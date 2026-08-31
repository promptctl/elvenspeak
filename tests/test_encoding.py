"""Encoder properties provable without a voice model.

Its own module because tests/test_api.py is gated on an installed Piper voice,
and this needs none — it feeds the encoder a buffer it constructs itself. Living
there meant the module-level skip removed it in exactly the case it was written
for: a CI machine with no baked-in model, where nothing else covers the claim
that an output format's name describes the bytes it returns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from elvenspeak.formats import OutputFormat
from elvenspeak.encoding import SynthesisFailed, encode, encode_stream

NATIVE_RATE = 22050
#: One second of a cheap non-silent waveform. Non-silent because some encoders
#: special-case digital silence, and a test that passes only on silence proves
#: less than it appears to.
ONE_SECOND = b"\x00\x20" * NATIVE_RATE


@pytest.mark.parametrize("name,rate", [("pcm_8000", 8000), ("pcm_16000", 16000),
                                       ("pcm_22050", 22050), ("pcm_44100", 44100)])
async def test_pcm_length_matches_its_declared_rate(name, rate):
    """The rate in a format's name is the rate of the samples returned.

    Raw PCM carries no header, so a caller handed the wrong rate cannot discover
    it — the audio just plays at the wrong pitch. That is the silent wrong answer
    this service exists to stop producing.

    One buffer encoded several ways, rather than several synthesis calls: Piper
    is not deterministic (the same sentence gave 36352, 37888 and 37376 samples
    across three runs, since VITS samples from a noise distribution), so
    comparing two synthesized responses could never be tighter than that
    variance — far too coarse to catch a rate that is subtly wrong.
    """
    encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse(name))
    assert len(encoded) / 2 / rate == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    "name,magic",
    [
        ("mp3_44100_128", b"\xff"),
        ("wav_22050", b"RIFF"),
        ("opus_48000_64", b"OggS"),
    ],
)
async def test_container_matches_the_format_asked_for(name, magic):
    encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse(name))
    assert encoded.startswith(magic)


async def test_mp3_carries_no_id3_tag():
    """ElevenLabs' MP3 begins at a frame sync; ffmpeg's would begin at a tag."""
    encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse("mp3_44100_128"))
    assert not encoded.startswith(b"ID3")


async def test_every_published_format_encodes():
    """No format in the table is one this server cannot actually produce.

    The table is transcribed from a published spec, so a row could name a codec
    ffmpeg was not built with — which would surface as a 500 on the first caller
    to ask for it rather than as anything visible here.
    """
    from elvenspeak.formats import SUPPORTED_OUTPUT_FORMATS

    for name in SUPPORTED_OUTPUT_FORMATS:
        encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse(name))
        assert encoded, f"{name} produced no bytes"


async def test_a_producer_failure_is_raised_not_encoded_as_a_short_answer():
    """[LAW:no-silent-failure] The failure this service was rebuilt to stop.

    When synthesis dies partway, ffmpeg encodes whatever it received and exits 0
    — a clean 200 carrying half an answer, indistinguishable from a short
    sentence. The pump's outcome is awaited and re-raised precisely so the
    encoder's exit status cannot speak for a producer it never saw.
    """
    def failing_chunks():
        yield ONE_SECOND
        raise RuntimeError("piper fell over mid-utterance")

    with pytest.raises(SynthesisFailed):
        await encode_stream_to_bytes(failing_chunks())


async def encode_stream_to_bytes(chunks):
    return b"".join(
        [part async for part in encode_stream(chunks, NATIVE_RATE, OutputFormat.parse("pcm_22050"))]
    )


async def test_a_failure_on_the_very_first_chunk_still_raises():
    """No output produced at all is the same lie, told with an empty body."""
    def failing_immediately():
        raise RuntimeError("piper never started")
        yield  # pragma: no cover - generator marker

    with pytest.raises(SynthesisFailed):
        await encode_stream_to_bytes(failing_immediately())


PACKAGE = Path(__file__).resolve().parent.parent / "elvenspeak"


def _imported_names(module: str) -> set[str]:
    """Every module `module` names in an import, at any nesting depth.

    `ast.walk` rather than a scan of the top level, because this package imports
    Piper exclusively from inside functions and `if TYPE_CHECKING` blocks — so a
    check that only read module-scope imports would report every module in the
    package as Piper-free, including the ones built entirely around it.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse((PACKAGE / f"{module}.py").read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `from .formats import X` names the module; `from . import speech`
            # names it in the alias list instead.
            found.update(
                {node.module} if node.module else {a.name for a in node.names}
            )
    return found


@pytest.mark.parametrize("root", ["encoding", "text"])
def test_the_encoder_cannot_reach_the_engine(root: str):
    """The seam, asserted rather than described.

    Encoding is engine-agnostic for every engine that will ever exist — they all
    emit PCM and ffmpeg does not care who produced it. That claim is what makes
    the encoder the reusable half, and it is worth nothing as a sentence in a
    docstring: one convenience import of a Piper type for an annotation would
    quietly undo it, every other test would stay green, and the second engine
    would discover the coupling instead of the reviewer.

    Read statically off the import graph rather than observed at runtime. The
    obvious runtime version — import the module and look for `piper` in
    `sys.modules` — passes whether or not the seam holds, because nothing here
    imports Piper at module scope; it was written, it went green, and it was
    only caught by re-coupling the module on purpose to watch it stay green.

    Transitive, because `from . import speech` inside `encoding` would satisfy a
    direct-import check while dragging the whole engine in behind it.
    """
    first_party = {path.stem for path in PACKAGE.glob("*.py")}
    seen: set[str] = set()
    queue = [root]

    while queue:
        module = queue.pop()
        seen.add(module)
        for name in _imported_names(module):
            top = name.split(".")[0]
            assert top != "piper", f"{module} imports {name}; reached from {root}"
            if top in first_party and top not in seen:
                queue.append(top)
