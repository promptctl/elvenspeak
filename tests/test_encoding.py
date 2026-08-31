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
from conftest import ENGINE_LIBRARIES

from elvenspeak.encoding import EncodingFailed, encode, encode_stream
from elvenspeak.formats import OutputFormat

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

    with pytest.raises(EncodingFailed):
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

    with pytest.raises(EncodingFailed):
        await encode_stream_to_bytes(failing_immediately())


PACKAGE = Path(__file__).resolve().parent.parent / "elvenspeak"
PACKAGE_NAME = PACKAGE.name


def _imported_modules(module_file: Path) -> set[str]:
    """Canonical dotted names of every module `module_file` imports.

    Canonical because one module has four spellings — `import elvenspeak.piper`,
    `from elvenspeak import piper`, `from .piper import PiperEngine`,
    `from . import piper` — and a matcher that recognises only the spellings the
    package happens to use today has a hole exactly the shape of the ones it
    does not. Every form is resolved to `elvenspeak.piper` here, once, so that
    neither the engine check nor the graph walk below has to know how an import
    was written.

    `ast.walk` rather than a scan of the top level, because this package imports
    Piper exclusively from inside functions and `if TYPE_CHECKING` blocks — so a
    check that only read module-scope imports would report every module in the
    package as Piper-free, including the ones built entirely around it.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(module_file.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A flat package has no `..` to resolve against, and guessing at one
            # would silently mis-resolve rather than fail. [LAW:no-silent-failure]
            assert node.level <= 1, f"{module_file}: unhandled relative import"
            base = node.module or ""
            if node.level:
                base = f"{PACKAGE_NAME}.{base}" if base else PACKAGE_NAME
            # Both, because `from x import y` names a module in `x` when y is a
            # submodule and a class in `x` when it is not; the resolver below
            # tells them apart by which one is backed by a file.
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _followed_module_file(name: str) -> Path | None:
    """The file to walk into for a dotted name, or None if this graph skips it.

    The package root is skipped, and skipped deliberately: `__init__.py`
    re-exports the package, so treating it as a dependency of one of its own
    members makes the graph cyclic — every module reaches every other, and the
    seam check reports coupling that no code has. `from . import formats` names
    the root as well as the submodule, which is how a harmless sibling import
    ends up looking like a route to the engine.

    The cost is a literal `import elvenspeak` inside a package module, which
    would go unfollowed. Nothing writes that, and following it would make this
    check meaningless for every module in the package, so it is a limit taken on
    purpose rather than an oversight.
    """
    if name == PACKAGE_NAME or not name.startswith(f"{PACKAGE_NAME}."):
        return None
    path = PACKAGE / (name[len(PACKAGE_NAME) + 1 :].replace(".", "/") + ".py")
    return path if path.exists() else None


def _modules_reaching_an_engine(root: str) -> set[str]:
    """First-party modules reachable from `root` that name an engine library."""
    hits: set[str] = set()
    seen: set[str] = set()
    queue = [f"{PACKAGE_NAME}.{root}"]

    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        source = _followed_module_file(name)
        if source is None:
            continue
        for imported in _imported_modules(source):
            if imported.split(".")[0] in ENGINE_LIBRARIES:
                hits.add(name)
            elif _followed_module_file(imported) is not None:
                queue.append(imported)
    return hits


@pytest.mark.parametrize(
    "root",
    [
        "api",
        "voices",
        "alignment",
        "encoding",
        "formats",
        "text",
        "engine",
        # Added when engine selection became a value. `settings` is the one an
        # obvious design gets wrong: putting the name-to-engine lookup behind
        # `from_env` would make `api` — which imports this module for its API key
        # — reach every engine's third-party library, and the reusable half of
        # this package would stop being importable without them. `provisioning`
        # is the seam that selection crosses and must stay as engine-free as
        # `engine` itself.
        "settings",
        "provisioning",
    ],
)
def test_the_server_cannot_reach_a_concrete_engine(root: str):
    """The seam, asserted rather than described.

    Every module named here is part of the ElevenLabs surface — the reusable
    half — and holds only [`elvenspeak.engine`]'s vocabulary. That claim is worth
    nothing as a sentence in a docstring: one convenience import of a Piper type
    for an annotation would quietly undo it, every other test would stay green,
    and the second engine would discover the coupling instead of the reviewer.

    Read statically off the import graph rather than observed at runtime. The
    obvious runtime version — import the module and look for `piper` in
    `sys.modules` — passes whether or not the seam holds, because nothing here
    imports an engine library at module scope; it was written, it went green, and
    it was only caught by re-coupling the module on purpose to watch it stay
    green.

    Transitive, because `from . import piper` inside `api` would satisfy a
    direct-import check while dragging the whole engine in behind it.
    """
    assert not _modules_reaching_an_engine(root)


@pytest.mark.parametrize("root", ["piper", "kokoro"])
def test_the_seam_check_can_actually_fail(root: str):
    """Positive control: the detector still detects.

    A test that cannot fail proves nothing, and this one has now been unable to
    fail twice — first as a runtime `sys.modules` probe that saw nothing because
    Piper is imported lazily, then as a matcher that followed relative imports
    only and would have walked straight past `import elvenspeak.piper`. Both were
    green, and both were caught by hand rather than by the suite.

    `piper` genuinely does reach an engine library, so it pins the detector from
    the other side: any future edit to the resolver that quietly stops finding
    things turns this red instead of turning the seam test vacuous. Every engine
    added to `conftest.ENGINE_LIBRARIES` belongs in this parametrize list too.
    """
    assert _modules_reaching_an_engine(root)
