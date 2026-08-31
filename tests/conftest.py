"""Fixtures and stand-ins shared by more than one test module.

[LAW:one-source-of-truth] `make_voice` lives here because two files now need a
voice on disk that no real download produced, and a second copy of "what a
`.onnx.json` has to contain" would be free to drift from the first — leaving one
file testing against a sidecar shape the other has stopped believing in.

`DeclaredEngine` and `needs_installed_model` are here for the same reason from
the two other directions a test reaches an engine: the fake one every capability
test drives, and where the real one's model is kept.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from elvenspeak.engine import (
    Capability,
    Prosody,
    Speech,
    TimedSpeech,
    Timing,
    Voice,
)

#: Every variable a startup reads — the server's own and the engines' — since
#: `Settings.from_env` now splices an engine's parse into its own. One copy,
#: because a second one is free to drift: the next setting added would be
#: remembered in one test's clearing list and forgotten in the other, and the
#: test that forgot goes flaky later with nothing pointing at the cause.
_ENVIRONMENT = (
    "ELVENSPEAK_ENGINE",
    "ELVENSPEAK_FALLBACK_VOICE",
    "ELVENSPEAK_API_KEY",
    "ELVENSPEAK_TIMESTAMPS",
    "PIPER_VOICES",
    "PIPER_MODELS_DIR",
    "PIPER_ALLOW_DOWNLOAD",
    "HOST",
    "PORT",
)


#: The voice that a test wanting real audio needs installed, and where the image
#: build and the developer loop both put it. One copy, because three files now
#: skip themselves when it is missing and three spellings of "is the model there"
#: are free to disagree about which model or which directory — leaving one file
#: quietly testing nothing while the others ran.
INSTALLED_VOICE = "en_US-lessac-medium"
MODELS_DIR = Path(
    os.environ.get("PIPER_MODELS_DIR", Path(__file__).parent.parent / "models")
)

#: Applied by the tests that synthesize for real. Skipped rather than mocked: a
#: mocked Piper proves the caller calls something, and what these tests are for
#: is what a real model actually produces.
needs_installed_model = pytest.mark.skipif(
    not (MODELS_DIR / f"{INSTALLED_VOICE}.onnx").exists(),
    reason=f"no {INSTALLED_VOICE} model in {MODELS_DIR}; "
    "fetch it with `uv run python -m elvenspeak.bake`",
)


def piper_prepared(
    models_dir: Path = MODELS_DIR,
    *,
    voices: tuple[str, ...] = (INSTALLED_VOICE,),
    allow_download: bool = False,
    timings: bool = False,
):
    """Piper configured for a test, through the door a deployment uses.

    [LAW:behavior-not-structure] Four files want a Piper engine over a chosen
    directory, and every one of them goes through `piper.configure` rather than
    reaching past it. A test that built the engine some other way would keep
    passing after the parse it skipped stopped being able to produce that value —
    which is the whole guarantee the parse exists to give.
    """
    from elvenspeak import piper

    return piper.configure(
        {
            "PIPER_VOICES": ",".join(voices),
            "PIPER_MODELS_DIR": str(models_dir),
            "PIPER_ALLOW_DOWNLOAD": "1" if allow_download else "0",
            "ELVENSPEAK_TIMESTAMPS": "1" if timings else "0",
        }
    )

#: Two, so that "every voice the engine offers" is a claim about more than one.
DECLARED_VOICES = (
    Voice(id="fake-voice", name="Fake", description="a test engine's voice"),
    Voice(id="fake-voice-two", name="Fake Two", description="its other voice"),
)

#: Audio per character of text, at [`_DECLARED_RATE`]. Arbitrary: what matters is
#: that it is a positive constant, which is what makes a longer text audibly
#: longer without the engine having to model anything about speech.
_SAMPLES_PER_CHARACTER = 100
_DECLARED_RATE = 22050


class DeclaredEngine:
    """An engine that does exactly what it was told to declare, and nothing more.

    Not a mock of Piper. It declares an arbitrary capability set and makes an
    unconvincing noise, which is the shape of the second engine this seam exists
    for: something written elsewhere, never anticipated here, and described
    accurately anyway.

    [LAW:one-type-per-behavior] One type taking a capability set, rather than a
    `SpeedlessEngine` and a `TimelessEngine` beside it. What differs between the
    engines these tests need is a value, so it is passed as one — which is the
    same argument the interface itself makes, tested here by being relied upon.

    Honest in both directions, which is what lets it stand as a subject of the
    conformance suite rather than only as a foil for the API's headers: what it
    declares, it really does, and what it does not declare, it really refuses.
    """

    def __init__(self, capabilities: frozenset[Capability]) -> None:
        self._capabilities = capabilities

    def voices(self) -> tuple[Voice, ...]:
        return DECLARED_VOICES

    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        return Speech(
            sample_rate=_DECLARED_RATE, audio=_silence(self._length(text, prosody))
        )

    def speak_timed(self, voice: Voice, text: str, prosody: Prosody) -> TimedSpeech:
        """One stretch covering the whole utterance, for an engine that measures.

        [LAW:no-silent-failure] Refuses outright without the capability, rather
        than returning something plausible. The server promises never to ask —
        the timestamp endpoints answer 501 first — and a stand-in that quietly
        obliged anyway would leave that promise resting on a gate no test failure
        would ever be traced back to.
        """
        if Capability.TIMESTAMPS not in self._capabilities:
            raise AssertionError(
                "speak_timed was called on an engine that did not declare "
                f"{Capability.TIMESTAMPS.name}"
            )
        samples = self._length(text, prosody)
        return TimedSpeech(
            pcm=b"".join(_silence(samples)),
            sample_rate=_DECLARED_RATE,
            timings=(Timing(samples=samples, separates_words=False),),
        )

    def _length(self, text: str, prosody: Prosody) -> int:
        """How many samples this utterance runs to.

        `prosody.speed` is read only when [`Capability.SPEED`] was declared. An
        engine that applied a speed it never claimed would be the dishonest one
        the ignored header cannot describe — and reading it unconditionally here
        would make this stand-in that engine, silently, for every test that
        constructs it declaring nothing.
        """
        speed = prosody.speed if Capability.SPEED in self._capabilities else 1.0
        return int(len(text) * _SAMPLES_PER_CHARACTER / speed)


@dataclass(frozen=True)
class DeclaredPrepared:
    """[`DeclaredEngine`] as something a deployment could have chosen.

    Satisfies `elvenspeak.provisioning.Prepared` in the smallest honest way: a
    `Settings` needs one, and a test that is about the API surface should not
    have to own a models directory to get it. It is also the worked example of
    the second half of what supplying an engine costs — nothing beyond the two
    methods, no configuration, no assets to install.
    """

    capabilities: frozenset[Capability] = frozenset(Capability)

    def acquire(self) -> tuple[Voice, ...]:
        """Nothing to install, which an engine with no assets says by saying so.

        No voices rather than the two it will serve: this engine makes its noise
        in memory, so a build has nothing to put on disk for it and nothing to
        prove it put there. That is the case `provisioning.Prepared.acquire`
        describes for a remote API, and returning the served voices here would
        leave the only worked example of the assetless path modelling it wrongly.
        """
        return ()

    def open(self) -> DeclaredEngine:
        return DeclaredEngine(self.capabilities)


def _silence(samples: int) -> Iterator[bytes]:
    """`samples` of quiet, in more than one piece.

    Split because a single chunk would let a consumer that mishandles chunk
    boundaries pass — the streaming encoder is pumped from this iterator, and one
    chunk is the one case where there are no boundaries to get wrong.
    """
    half = samples // 2
    yield b"\x00\x00" * half
    yield b"\x00\x00" * (samples - half)


@pytest.fixture
def clean_env(monkeypatch):
    """An environment holding nothing this service reads.

    For the tests that call `from_env` with no argument — the process-facing
    path, which cannot be handed a dict. Without this they inherit whatever the
    shell exports, and an auto-injected `PORT` or a leftover
    `ELVENSPEAK_TIMESTAMPS` fails them for reasons unrelated to their subject.
    """
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def make_voice(
    models_dir: Path, key: str = "en_US-lessac-medium", sample_rate: int = 22050
) -> None:
    """A voice as far as installing and opening can tell: both halves present.

    No real Piper model is needed. `_describe` reads only the `.onnx.json`
    sidecar and never opens the weights, so the `.onnx` here is a placeholder
    whose only job is to exist.

    Every field the key states is read back out of it, rather than fixed at the
    values its first caller happened to use. A sidecar saying `en_US` for a key
    saying `en_GB` is a fixture contradicting itself, and it costs nothing until
    the first test to assert on a language label gets a wrong answer from the
    thing it was trusting to be right.

    The unpack is the check: a key that is not `<lang>-<name>-<quality>` fails
    here and says so, rather than being written to disk as a voice.
    """
    language, dataset, quality = key.split("-")
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{key}.onnx").write_bytes(b"not a real model")
    (models_dir / f"{key}.onnx.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "language": {"code": language},
                "audio": {"sample_rate": sample_rate, "quality": quality},
                "num_speakers": 1,
            }
        ),
        encoding="utf-8",
    )
