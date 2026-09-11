"""The official ElevenLabs SDK, driving a real elvenspeak server over a socket.

Every other test in this suite reaches this API through a client this repo wrote
against its own reading of ElevenLabs' API, so the suite and the server share one
author's misunderstanding and cannot disagree with each other. This file is the
one place a disagreement is possible: the requests are composed by ElevenLabs'
own client and the answers are parsed by ElevenLabs' own response models.

Real uvicorn on a loopback socket rather than `TestClient`, because the SDK
builds its own HTTP and a transport that shortcut it would be testing a
different program ([LAW:behavior-not-structure] — the contract is what a foreign
client observes over the wire). What stays fake is only the engine behind the
server, the same [`DeclaredEngine`] seam every other conformance test stands on.

WHAT "THE SDK PARSED IT" DOES NOT PROVE, and why every test here asserts past
it. The SDK's response models are far looser than this server's promises:
`Voice` requires only `voice_id` and holds twenty-five fields optional, `Model`
requires only `model_id`, and `AudioWithTimestampsResponse` holds *both*
`alignment` and `normalized_alignment` optional. A server answering
`{"voices": [{"voice_id": "x"}]}` would satisfy every one of those models. So a
test asserting no more than "it parsed" is a check that cannot fail, which is
the defect this epic exists to remove rather than to reproduce
([LAW:verifiable-goals] — done needs a shape that failure can miss). Each test
below names the fields this server promises that the SDK itself would shrug at,
and those assertions are where a regression lands.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from conftest import DECLARED_VOICES
from elevenlabs.client import ElevenLabs
from fleet import engine_app, serving

from elvenspeak import formats

#: Configured on the server and handed to the SDK, so every call below also
#: proves the two agree on the header carrying it. The SDK sends `xi-api-key`
#: and `api.require_key` reads `xi-api-key`; a rename on either side breaks
#: every real client, and with no key configured nothing here would notice.
_API_KEY = "sdk-conformance-key"


@pytest.fixture(scope="module")
def client() -> Iterator[ElevenLabs]:
    """The official client, pointed at a real elvenspeak deployment.

    Module-scoped: every call this file makes is a read, so one server serves
    them all, and booting one per test would buy no isolation
    ([LAW:carrying-cost]).
    """
    with serving(engine_app("piper", DECLARED_VOICES, api_key=_API_KEY)) as base_url:
        yield ElevenLabs(base_url=base_url, api_key=_API_KEY)


@pytest.fixture(scope="module")
def voice_id(client: ElevenLabs) -> str:
    """A voice this deployment offers, discovered the way a real client must.

    Taken by position rather than by capability because the SDK's `Voice` model
    drops elvenspeak's `capabilities` extension entirely — an SDK-only client
    cannot see it. Position is sound here because `engine_app` gives every
    declared voice the full capability set; it is not a claim that the first
    voice listed is special. It is not: `voices.py` sorts the listing
    alphabetically while the fallback is chosen in the engine's own order.
    """
    return client.voices.get_all().voices[0].voice_id


def test_the_sdk_reads_every_voice_field_this_server_promises(client: ElevenLabs) -> None:
    """`name` and `category` are optional to the SDK and promised by this server.

    Dropping either would leave `voices.get_all()` parsing cleanly, so the
    assertions are on the fields rather than on the parse.
    """
    listing = client.voices.get_all()

    assert listing.voices, "a deployment that offers no voices cannot be a drop-in"
    for voice in listing.voices:
        assert voice.voice_id, "every voice is addressable by id"
        assert voice.name, f"{voice.voice_id} came back with no name"
        assert voice.category, f"{voice.voice_id} came back with no category"


def test_the_sdk_reads_the_models_this_deployment_serves(client: ElevenLabs) -> None:
    """`models.list`, not `models.get_all` — the latter does not exist in v2.

    `name` and `can_do_text_to_speech` are optional to the SDK's `Model`. A
    text-to-speech deployment that answered `can_do_text_to_speech: false` would
    still parse, and would be telling every client not to call it.
    """
    served = client.models.list()

    assert served, "a deployment that names no model cannot answer model_id"
    for model in served:
        assert model.model_id, "every model is addressable by id"
        assert model.name, f"{model.model_id} came back with no name"
        assert model.can_do_text_to_speech, (
            f"{model.model_id} is served by a text-to-speech deployment and says "
            "it cannot do text to speech"
        )


def test_convert_returns_audio_the_sdk_can_assemble(
    client: ElevenLabs, voice_id: str
) -> None:
    """The SDK streams `convert` as chunks; the bytes must be real audio.

    The expected codec is read off `formats.DEFAULT_OUTPUT_FORMAT` rather than
    written here, so this test cannot disagree with the module that owns the
    default ([LAW:one-source-of-truth]).
    """
    assert formats.DEFAULT_OUTPUT_FORMAT.startswith("mp3"), (
        "this test reads the default format's codec off formats.py; the default "
        f"is now {formats.DEFAULT_OUTPUT_FORMAT} and the assertion below is stale"
    )

    audio = b"".join(client.text_to_speech.convert(voice_id=voice_id, text="hello there"))

    assert _is_mp3(audio), f"convert returned {len(audio)} bytes that are not mp3"


def test_stream_returns_audio_through_the_streaming_path(
    client: ElevenLabs, voice_id: str
) -> None:
    """The SDK's streaming path is a distinct endpoint, and must answer in kind.

    A deployment can serve `convert` and leave `stream` broken; a real client
    that streams would then fail against a server this suite called green.
    """
    audio = b"".join(client.text_to_speech.stream(voice_id=voice_id, text="streaming now"))

    assert _is_mp3(audio), f"stream returned {len(audio)} bytes that are not mp3"


def test_with_timestamps_carries_the_alignment_the_sdk_holds_optional(
    client: ElevenLabs, voice_id: str
) -> None:
    """Both alignments are optional to the SDK, so parsing proves nothing here.

    This is the sharpest instance of the file's premise: a server that stopped
    returning alignment entirely would still hand back a valid
    `AudioWithTimestampsResponse`, and a client asking for timestamps would get
    an object with none in it. The per-character arrays are asserted to be the
    same length as the characters they time, because ragged arrays are the shape
    a real alignment regression takes.
    """
    spoken = "hi there"

    answer = client.text_to_speech.convert_with_timestamps(voice_id=voice_id, text=spoken)

    assert answer.audio_base_64, "no audio came back with the timestamps"
    for label, alignment in (
        ("alignment", answer.alignment),
        ("normalized_alignment", answer.normalized_alignment),
    ):
        assert alignment is not None, (
            f"this deployment declares timestamps and returned no {label}"
        )
        starts = alignment.character_start_times_seconds
        ends = alignment.character_end_times_seconds
        assert len(alignment.characters) == len(starts) == len(ends), (
            f"{label} timed {len(starts)} starts and {len(ends)} ends against "
            f"{len(alignment.characters)} characters"
        )
        assert starts == sorted(starts), f"{label} runs backwards in time"


def _is_mp3(audio: bytes) -> bool:
    """Whether `audio` opens on an MP3 frame sync — eleven set bits.

    Enough to tell real frames from an empty body, an error page, or silence
    encoded as a different codec, which are the ways this can regress. Not a
    decoder: proving the audio is *correct* is `test_encoding`'s job, and
    repeating it here would be a second opinion that could drift from it.
    """
    return len(audio) > 2 and audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0
