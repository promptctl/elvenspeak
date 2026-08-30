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


def test_non_streaming_endpoint_returns_encoded_audio(client):
    """The primary one-shot endpoint, previously never called by any test.

    `speak()`'s `path=""` default existed and was never used — every call passed
    "/stream" — so nothing asserted this handler returned anything at all.
    """
    response = speak(client, output_format="mp3_44100_128")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content.startswith(b"\xff")


def test_default_settings_have_the_elevenlabs_shape(client):
    body = client.get("/v1/voices/settings/default").json()
    assert body["speed"] == 1.0
    assert set(body) == {
        "stability", "similarity_boost", "style", "use_speaker_boost", "speed",
    }


def test_per_voice_settings_answer_for_installed_voices_only(client):
    assert client.get(f"/v1/voices/{VOICE}/settings").json()["speed"] == 1.0
    assert client.get(f"/v1/voices/{FOREIGN_ID}/settings").status_code == 404


def test_empty_text_is_refused_before_synthesis(client):
    """A streaming endpoint cannot report this later — the 200 is already sent."""
    for path in ("", "/stream", "/stream/with-timestamps"):
        response = client.post(
            f"/v1/text-to-speech/{VOICE}{path}", json={"text": ""}
        )
        assert response.status_code == 422, path


def test_oversized_text_is_refused(client):
    from elvenspeak.api import MAX_TEXT_LENGTH

    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream",
        json={"text": "a" * (MAX_TEXT_LENGTH + 1)},
    )
    assert response.status_code == 422


def test_unknown_body_fields_are_reported_not_dropped(client):
    """Rule 2, applied to parameters this server has never heard of."""
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream",
        json={"text": "hello", "some_future_elevenlabs_field": 3},
    )
    assert response.status_code == 200
    assert "some_future_elevenlabs_field" in response.headers["x-elvenspeak-ignored"]


def test_streamed_timestamps_report_fidelity_per_object(client):
    """Fidelity varies per sentence, so it cannot live in a single header."""
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream/with-timestamps",
        json={"text": "First sentence here. Second sentence follows."},
    )
    objects = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert objects
    for obj in objects:
        assert obj["alignment_fidelity"] in {"word-exact", "interpolated"}


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
