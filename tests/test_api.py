"""The HTTP surface, against a real voice.

Skipped rather than mocked when no model is installed. A mocked Piper would
prove the handlers call something, which is not the property under test — what
matters is that the bytes coming back are in the format the caller asked for,
and only a real encode can show that.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elvenspeak import create_app
from elvenspeak.settings import Settings

VOICE = "en_US-lessac-medium"
MODELS = Path(os.environ.get("PIPER_MODELS_DIR", Path(__file__).parent.parent / "models"))

pytestmark = pytest.mark.skipif(
    not (MODELS / f"{VOICE}.onnx").exists(),
    reason=f"no {VOICE} model in {MODELS}; run scripts/fetch-voice.sh",
)

#: An ElevenLabs voice id from aliases.toml. Used to prove substitution, which
#: is the behaviour openconv depends on.
FOREIGN_ID = "21m00Tcm4TlvDq8ikWAM"


@pytest.fixture(scope="module")
def client():
    settings = Settings(
        voices=(VOICE,),
        fallback=VOICE,
        models_dir=MODELS,
        allow_download=False,
        api_key=None,
        timestamps=True,
        host="127.0.0.1",
        port=0,
    )
    with TestClient(create_app(settings)) as started:
        yield started


def speak(client, path="", **params):
    return client.post(
        f"/v1/text-to-speech/{VOICE}{path}",
        json={"text": "Compatibility is measurable."},
        params=params,
    )


def test_health_lists_installed_voices(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert VOICE in body["voices"]


@pytest.mark.parametrize(
    "fmt,content_type,magic",
    [
        ("mp3_44100_128", "audio/mpeg", b"\xff"),
        ("wav_22050", "audio/wav", b"RIFF"),
        ("opus_48000_64", "audio/ogg", b"OggS"),
    ],
)
def test_output_format_is_honoured(client, fmt, content_type, magic):
    """The regression this whole rewrite exists to prevent."""
    response = speak(client, "/stream", output_format=fmt)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)
    assert response.content.startswith(magic)


def test_mp3_has_no_id3_tag(client):
    """ElevenLabs' MP3 starts at a frame sync; a tag would shift every offset."""
    body = speak(client, "/stream", output_format="mp3_44100_128").content
    assert not body.startswith(b"ID3")
    assert body[0] == 0xFF


async def test_pcm_length_matches_its_declared_rate():
    """Proof the rate in a format's name is the rate of the samples returned.

    Raw PCM carries no header, so a caller handed the wrong rate cannot discover
    it — the audio just plays at the wrong pitch, which is the silent-wrong-
    answer this service is being rebuilt to stop producing.

    Tested against the encoder directly rather than through the API, because
    Piper is not deterministic: the same sentence synthesized twice differs by a
    few percent in length (measured: 36352, 37888, 37376 samples across three
    runs of one text), since a VITS model samples from a noise distribution.
    Two HTTP calls could therefore never be compared to better than that
    variance, which is far too coarse to catch a rate that is subtly wrong. One
    buffer encoded several ways holds everything constant but the thing under
    test.
    """
    from elvenspeak.formats import OutputFormat
    from elvenspeak.speech import encode

    native_rate = 22050
    one_second = b"\x00\x01" * native_rate

    durations = {}
    for name, rate in (("pcm_8000", 8000), ("pcm_16000", 16000), ("pcm_44100", 44100)):
        encoded = await encode(one_second, native_rate, OutputFormat.parse(name))
        durations[name] = len(encoded) / 2 / rate

    for name, seconds in durations.items():
        assert seconds == pytest.approx(1.0, abs=0.01), f"{name} came back {seconds}s"


def test_unknown_format_is_refused_with_the_offending_value(client):
    response = speak(client, "/stream", output_format="flac_44100")
    assert response.status_code == 422
    assert "flac_44100" in json.dumps(response.json())


def test_unknown_voice_substitutes_and_says_so(client):
    """The contract openconv depends on, and the header that keeps it honest."""
    response = client.post(
        f"/v1/text-to-speech/{FOREIGN_ID}/stream", json={"text": "hello"}
    )
    assert response.status_code == 200
    assert response.headers["x-elvenspeak-voice"] == VOICE
    assert response.headers["x-elvenspeak-voice-requested"] == FOREIGN_ID


def test_known_voice_reports_no_substitution(client):
    response = speak(client, "/stream")
    assert response.headers["x-elvenspeak-voice"] == VOICE
    assert "x-elvenspeak-voice-requested" not in response.headers


def test_unhonourable_parameters_are_named_back(client):
    """[LAW:no-silent-failure] Accepted-and-dropped is reported, not hidden."""
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream",
        json={
            "text": "hello",
            "model_id": "eleven_multilingual_v2",
            "seed": 42,
            "voice_settings": {"stability": 0.3, "speed": 1.5},
        },
    )
    ignored = response.headers["x-elvenspeak-ignored"]
    assert "model_id" in ignored
    assert "seed" in ignored
    assert "voice_settings.stability" in ignored
    # speed IS honoured, so naming it would be a lie in the other direction.
    assert "speed" not in ignored.replace("voice_settings.stability", "")


def test_speed_actually_changes_the_audio(client):
    """Otherwise `speed` belongs in the ignored list with the rest."""
    def length(speed):
        body = client.post(
            f"/v1/text-to-speech/{VOICE}/stream",
            json={"text": "One two three four five.", "voice_settings": {"speed": speed}},
            params={"output_format": "pcm_22050"},
        ).content
        return len(body)

    assert length(2.0) < length(1.0) * 0.75


def test_discovery_does_not_substitute(client):
    """Synthesis answers for any id; listing must only report what it has."""
    assert client.get(f"/v1/voices/{FOREIGN_ID}").status_code == 404
    assert client.get(f"/v1/voices/{VOICE}").status_code == 200


def test_voices_listing_has_the_elevenlabs_shape(client):
    body = client.get("/v1/voices").json()
    assert [v["voice_id"] for v in body["voices"]] == [VOICE]
    assert body["voices"][0]["name"]
    assert "labels" in body["voices"][0]


def test_timestamps_cover_the_input_text(client):
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/with-timestamps",
        json={"text": "Hello there, friend."},
    )
    assert response.status_code == 200
    assert response.headers["x-elvenspeak-alignment"] == "word-exact"
    body = response.json()
    assert base64.b64decode(body["audio_base64"])
    alignment = body["alignment"]
    assert "".join(alignment["characters"]) == "Hello there, friend."
    assert alignment["character_end_times_seconds"][-1] > 0


def test_streamed_timestamps_form_one_continuous_timeline(client):
    """Each object picks up where the previous one left off."""
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream/with-timestamps",
        json={"text": "First sentence here. Second sentence follows."},
    )
    objects = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert len(objects) == 2
    first_end = objects[0]["alignment"]["character_end_times_seconds"][-1]
    second_start = objects[1]["alignment"]["character_start_times_seconds"][0]
    assert second_start >= first_end


def test_api_key_is_enforced_when_configured():
    settings = Settings(
        voices=(VOICE,),
        fallback=VOICE,
        models_dir=MODELS,
        allow_download=False,
        api_key="s3cret",
        timestamps=False,
        host="127.0.0.1",
        port=0,
    )
    with TestClient(create_app(settings)) as guarded:
        payload = {"text": "hello"}
        assert guarded.post(f"/v1/text-to-speech/{VOICE}/stream", json=payload).status_code == 401
        allowed = guarded.post(
            f"/v1/text-to-speech/{VOICE}/stream",
            json=payload,
            headers={"xi-api-key": "s3cret"},
        )
        assert allowed.status_code == 200
        # Health stays open so a load balancer does not need the credential.
        assert guarded.get("/health").status_code == 200


def test_timestamps_disabled_refuses_rather_than_inventing(client):
    """[LAW:no-silent-failure] 501 beats plausible numbers derived from nothing."""
    settings = Settings(
        voices=(VOICE,),
        fallback=VOICE,
        models_dir=MODELS,
        allow_download=False,
        api_key=None,
        timestamps=False,
        host="127.0.0.1",
        port=0,
    )
    with TestClient(create_app(settings)) as plain:
        response = plain.post(
            f"/v1/text-to-speech/{VOICE}/with-timestamps", json={"text": "hello"}
        )
        assert response.status_code == 501
