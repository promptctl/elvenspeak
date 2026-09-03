"""The HTTP surface, against a real voice.

Installed rather than mocked, and fetched rather than skipped. A mocked Piper
would prove the handlers call something, which is not the property under test —
what matters is that the bytes coming back are in the format the caller asked
for, and only a real encode can show that. A machine without the voice gets it
downloaded, because a skip reads as a pass in a summary and would withdraw this
whole module on exactly the machines least likely to have run it.
"""

from __future__ import annotations

import base64
import json

import pytest
from conftest import declaring, DECLARED_VOICES, DeclaredEngine
from conftest import INSTALLED_VOICE as VOICE
from conftest import MODELS_DIR as MODELS
from conftest import piper_prepared
from fastapi.testclient import TestClient

from elvenspeak import create_app
from elvenspeak import declarations as declarations_mod
from elvenspeak.declarations import model_ids
from elvenspeak.engine import Capability
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings

#: Every test here synthesizes with the real Piper voice, so the whole module
#: depends on the assets being installed rather than skipping when they are not.
pytestmark = pytest.mark.usefixtures("piper_installed")

#: An ElevenLabs voice id from `elvenspeak/aliases/piper.toml`. Used to prove
#: substitution, which is the behaviour openconv depends on.
FOREIGN_ID = "21m00Tcm4TlvDq8ikWAM"


def settings_for(timings: bool = True, **overrides) -> Settings:
    """A deployment's settings. `timings` is the engine's, which is why it is not
    an override: whether durations can be reported is fixed when the model is
    opened, so it has to be said to the engine and not to the server.

    [LAW:one-source-of-truth] `known_engines` is derived from whichever
    `engine_name` won rather than sitting beside it as its own override. In
    `from_env` both come out of one registry lookup, so a deployment missing its
    own engine from its own roster is a state production cannot reach — and a
    helper that let a caller name an engine without adding it to the roster
    would hand every test the job of remembering to say it twice.
    """
    fields = {
        "engine": piper_prepared(MODELS, voices=(VOICE,), timings=timings),
        "engine_name": "piper",
        "withheld": frozenset(),
        "fallback": VOICE,
        "api_key": None,
        "host": "127.0.0.1",
        "port": 0,
        **overrides,
    }
    return Settings(
        known_engines=frozenset(ENGINES) | {fields["engine_name"]}, **fields
    )


def served(settings: Settings) -> TestClient:
    """The app as `main.build()` assembles it: a real engine behind the surface.

    The engine is opened here rather than by `create_app`, which is the property
    under test as much as a convenience — the API surface is handed one and never
    names it, and never learns how it was configured either.
    """
    return TestClient(create_app(settings, settings.engine.open()))


@pytest.fixture(scope="module")
def client():
    with served(settings_for()) as started:
        yield started


def speak(client, path="", **params):
    return client.post(
        f"/v1/text-to-speech/{VOICE}{path}",
        json={"text": "Compatibility is measurable."},
        params=params,
    )


def test_health_lists_installed_voices(client):
    """A server that can speak says so in the status line and names the voices.

    The body carries no `status` field any more: it repeated the status line,
    which is one verdict with two representations free to disagree. What a
    checker needs is the code, and the code is what every checker now reads —
    `tests/test_router.py` covers the same endpoint answering 503.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert VOICE in response.json()["voices"]


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


def test_the_alias_table_loaded_is_the_one_the_engines_name_picks(
    tmp_path, monkeypatch
):
    """[LAW:one-source-of-truth] The name on the settings picks the table, here.

    Two tables claiming the same foreign id, so the answer says which one was
    read. `decoy` sends it to the voice that is also the fallback, which is the
    whole point: the three ways this can go wrong — the wrong table, no table at
    all, and a name that reaches no file — every one of them ends at the
    fallback, so a test that asserted substitution would pass under all of them.
    Only the correct table produces the second voice.

    That is the gap the test above cannot cover. It observes a fallback, and a
    fallback is what a correct table, a wrong table and an absent table all
    look like from outside.

    No real voice, because the discriminator is the wiring and not the audio:
    the request goes through the same `create_app` a deployment builds.
    """
    (tmp_path / "named.toml").write_text(
        f'[elevenlabs]\n"foreign-id" = "{DECLARED_VOICES[1].id}"\n', encoding="utf-8"
    )
    (tmp_path / "decoy.toml").write_text(
        f'[elevenlabs]\n"foreign-id" = "{DECLARED_VOICES[0].id}"\n', encoding="utf-8"
    )
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    app = create_app(
        settings_for(engine_name="named", fallback=DECLARED_VOICES[0].id),
        DeclaredEngine(declaring(frozenset(Capability))),
    )
    response = TestClient(app).post(
        "/v1/text-to-speech/foreign-id/stream", json={"text": "hello"}
    )

    assert response.status_code == 200
    assert response.headers["x-elvenspeak-voice"] == DECLARED_VOICES[1].id
    assert response.headers["x-elvenspeak-voice-requested"] == "foreign-id"


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
    # The absence is the contract, not an oversight. Fidelity is decided per
    # sentence and can differ between objects of one response, so a single header
    # could report only one of several answers — and would have to be sent before
    # any of them were known. A client is meant to read the per-object field, so
    # re-adding a blanket header here would quietly restore the wrong answer.
    assert "x-elvenspeak-alignment" not in response.headers
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


def test_models_listing_has_the_elevenlabs_shape(client):
    """A bare array, where `/v1/voices` above wraps its list in an object.

    The asymmetry is ElevenLabs' own and mirroring it is the whole job: a stock
    client indexes this response directly, and helpfully wrapping it in a
    `models` key for consistency with the endpoint above would break exactly the
    callers this surface exists for. Pinned here because it looks like an
    oversight, and the next reader to tidy it will find this test instead.
    """
    body = client.get("/v1/models").json()

    assert isinstance(body, list)
    assert [entry["model_id"] for entry in body] == [
        "piper",
        *sorted(model_ids("piper")),
    ]
    assert body[0]["can_do_text_to_speech"] is True


def test_the_listing_names_every_model_id_the_service_will_accept(client):
    """[LAW:one-source-of-truth] Advertised and accepted are one answer.

    The endpoint exists so a caller can find out what `model_id` values are legal
    without sending one and reading the status code. That is only true while the
    listing and the refusal read the same table, so this asks both: every id
    advertised is served, and the engine this deployment is not running is
    neither advertised nor served.

    Driven through `create_app` rather than against `models.Directory`, because
    two readers agreeing inside one object is not the property — the property is
    that the endpoint a caller reads and the gate a caller hits agree.
    """
    listed = [entry["model_id"] for entry in client.get("/v1/models").json()]

    for model_id in listed:
        response = client.post(
            f"/v1/text-to-speech/{VOICE}/stream",
            json={"text": "hello", "model_id": model_id},
        )
        assert response.status_code == 200, (model_id, response.text)
        # Served means it chose the engine that spoke, so naming it as ignored
        # would be the header lying about the one field this deployment did act
        # on.
        assert "model_id" not in response.headers.get("x-elvenspeak-ignored", "")

    # The other engine and everything it claims are absent from both sides.
    elsewhere = {"kokoro", *model_ids("kokoro")}
    assert not elsewhere & set(listed)


@pytest.mark.parametrize(
    "path", ["", "/stream", "/with-timestamps", "/stream/with-timestamps"]
)
def test_asking_for_an_engine_this_deployment_is_not_running_is_refused(client, path):
    """The guarantee: that engine, or a refusal, never silently the other one.

    Parametrized over every synthesis endpoint because the gate is a line in each
    handler rather than something the framework applies for them — an endpoint
    added later without it would answer in Piper for a caller who asked for
    Kokoro, and the response would look exactly like success.
    """
    response = client.post(
        f"/v1/text-to-speech/{VOICE}{path}",
        json={"text": "hello", "model_id": "kokoro"},
    )
    assert response.status_code == 422, (path, response.text)
    assert "kokoro" in response.text


def test_a_model_id_naming_no_engine_here_still_speaks(client):
    """A stock ElevenLabs client sends a `model_id` on its very first request.

    Refusing the ones this deployment maps to nothing would reject most real
    callers, so an unrecognised id is reported as ignored and the voice decides —
    which is what already happened when the field was omitted. The refusal above
    is for ids that name an engine, and this is the line between the two.
    """
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream",
        json={"text": "hello", "model_id": "eleven_turbo_v2"},
    )
    assert response.status_code == 200, response.text
    assert "model_id" in response.headers["x-elvenspeak-ignored"]


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
    with served(settings_for(timings=False, api_key="s3cret")) as guarded:
        payload = {"text": "hello"}
        assert guarded.post(f"/v1/text-to-speech/{VOICE}/stream", json=payload).status_code == 401
        allowed = guarded.post(
            f"/v1/text-to-speech/{VOICE}/stream",
            json=payload,
            headers={"xi-api-key": "s3cret"},
        )
        assert allowed.status_code == 200
        # The discovery endpoints are guarded too. What a deployment can speak
        # with is not public just because answering costs nothing — and this is
        # the half a new endpoint forgets, since it works either way in every
        # test that does not set a key.
        assert guarded.get("/v1/models").status_code == 401
        assert (
            guarded.get("/v1/models", headers={"xi-api-key": "s3cret"}).status_code
            == 200
        )
        # Health stays open so a load balancer does not need the credential.
        assert guarded.get("/health").status_code == 200


def test_timestamps_disabled_refuses_rather_than_inventing(client):
    """[LAW:no-silent-failure] 501 beats plausible numbers derived from nothing."""
    with served(settings_for(timings=False)) as plain:
        response = plain.post(
            f"/v1/text-to-speech/{VOICE}/with-timestamps", json={"text": "hello"}
        )
        assert response.status_code == 501


@pytest.mark.parametrize("text", ["   ", "\t\n ", " "])
def test_text_with_nothing_to_say_is_refused(client, text):
    """422, not a 200 carrying no audio.

    `min_length=1` counts characters, so whitespace passed it. On the streaming
    timestamp endpoint that produced the worst available answer: `split_sentences`
    finds no sentences, the generator yields nothing, and the caller receives a
    successful, empty response for a request that was never going to work.
    """
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream/with-timestamps", json={"text": text}
    )
    assert response.status_code == 422


def test_text_is_spoken_as_stripped(client):
    """The alignment describes the text that was actually synthesized."""
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/with-timestamps", json={"text": "  hello  "}
    )
    assert response.status_code == 200
    assert "".join(response.json()["alignment"]["characters"]) == "hello"


def test_a_non_latin1_voice_id_still_answers(client):
    """The substitution contract, for an id that cannot be a header value.

    Header values are latin-1 on the wire, so echoing the requested id verbatim
    raised `UnicodeEncodeError` while building the response — turning the
    documented "unknown voice still gets audio" path into a 500 for any caller
    whose id happened to be non-Latin. The id is escaped rather than dropped, so
    the header still reports what was asked for.
    """
    response = client.post(
        "/v1/text-to-speech/日本語/stream", json={"text": "hello"}
    )
    assert response.status_code == 200
    assert response.headers["x-elvenspeak-voice"] == VOICE
    assert "65e5" in response.headers["x-elvenspeak-voice-requested"]


def test_a_non_latin1_body_field_name_still_answers(client):
    """The same wire limit, reached through `ignored()` rather than the URL."""
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream", json={"text": "hello", "日": 1}
    )
    assert response.status_code == 200
    assert "65e5" in response.headers["x-elvenspeak-ignored"]


def test_unknown_voice_settings_are_reported_not_dropped(client):
    """Rule 2, one level down from the body.

    Enumerating the four settings ElevenLabs publishes today made the promise
    true only for those four; a setting added later was discarded by the parser
    with nothing anywhere reporting it — the same silent drop the top-level
    `extra="allow"` was added to close.
    """
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream",
        json={"text": "hello", "voice_settings": {"some_future_setting": 1}},
    )
    assert response.status_code == 200
    assert "voice_settings.some_future_setting" in response.headers["x-elvenspeak-ignored"]


@pytest.mark.parametrize("escaped,raw", [
    ("%0D%0AX-Injected:%20evil", "\r"),
    ("%0AX-Injected:%20evil", "\n"),
    ("%00bar", "\x00"),
])
def test_control_characters_cannot_reach_a_header_value(client, escaped, raw):
    """CR and LF are ASCII, so escaping only what fails an ASCII encode let them
    through untouched — and a `voice_id` reaches the handler already
    percent-decoded, so `%0D%0A` in the URL became a real CRLF in a header value.

    Sent percent-encoded because that is the actual shape of the attack: an HTTP
    client will not put a raw control character in a request line, and the router
    decodes it before the handler ever sees it.

    Asserted on the emitted value rather than on whether the server below would
    have refused it — leaning on that unstated is the gap, not the mitigation.
    """
    response = client.post(
        f"/v1/text-to-speech/voice{escaped}/stream", json={"text": "hello"}
    )
    assert response.status_code == 200
    echoed = response.headers["x-elvenspeak-voice-requested"]
    assert raw not in echoed
    assert "X-Injected" not in response.headers


def test_ascii_safe_escapes_everything_outside_the_printable_range():
    """The rule stated directly, without a client in the way.

    Pinned as a property rather than as three examples: the previous version
    escaped on encodability, which silently exempted every ASCII control
    character, and only a range check makes the docstring's claim true.
    """
    from elvenspeak.api import _ascii_safe

    for codepoint in list(range(0x00, 0x20)) + [0x7F]:
        assert chr(codepoint) not in _ascii_safe(chr(codepoint))
    assert _ascii_safe("plain-voice_id.42") == "plain-voice_id.42"
    assert "65e5" in _ascii_safe("日")


def test_one_voice_serves_concurrent_requests():
    """Synthesis runs on worker threads against a single cached PiperVoice.

    Piper serializes espeak-ng behind its own module-level lock and ONNX Runtime
    supports concurrent Run() on one session, so no lock is needed here — but
    that is a claim about someone else's code, and this is what makes it checked
    rather than asserted. A regression would show up as an exception or as empty
    audio from one of the two calls.

    Byte equality is not asserted: Piper samples from a noise distribution, so
    the same sentence differs run to run. Length and non-emptiness are the
    properties that separate corruption from that ordinary variance.
    """
    from concurrent.futures import ThreadPoolExecutor

    from elvenspeak.engine import Prosody

    engine = piper_prepared(MODELS, voices=(VOICE,)).open()
    voice = engine.voices()[0]
    text = "The quick brown fox jumps over the lazy dog."

    def synth():
        return b"".join(engine.speak(voice, text, Prosody(speed=1.0)).audio)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [f.result() for f in [pool.submit(synth) for _ in range(4)]]

    assert all(len(pcm) > 0 for pcm in results)
    # Same sentence, same voice: lengths vary with sampling but not by orders of
    # magnitude. A corrupted or truncated concurrent run shows up here.
    shortest, longest = min(map(len, results)), max(map(len, results))
    assert longest < shortest * 2


def test_an_ignored_field_name_containing_a_comma_stays_one_name(client):
    """The separator cannot be forged by a field name.

    `extra="allow"` is what makes rule 2 hold for fields this server has never
    heard of, and it accepts a field literally named `a, b`. Joined bare, that
    name arrives looking like two ignored fields — the header misreporting the
    one thing it exists to report accurately.
    """
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream",
        json={"text": "hello", "a, b": 1},
    )
    assert response.status_code == 200
    reported = response.headers["x-elvenspeak-ignored"]
    assert "a\\x2cb" in reported.replace(" ", "")
    # One name, so splitting on the separator yields one entry.
    assert len([part for part in reported.split(", ") if part]) == 1


def test_the_alignment_header_goes_through_the_same_escaping(client):
    """Endpoint-specific headers are built by `headers()`, not assigned after it.

    This one is a closed enum today, so the value is safe either way; what is
    pinned is that it is routed through the single checkpoint, since a header
    assigned onto the result afterwards is how the next one would skip it.
    """
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/with-timestamps", json={"text": "hello"}
    )
    assert response.status_code == 200
    assert response.headers["x-elvenspeak-alignment"] in ("word-exact", "interpolated")
    assert response.headers["x-elvenspeak-voice"] == VOICE


def test_the_escaping_is_reversible(client):
    """A backslash the caller sent cannot pass as one the server produced.

    Every escape this function emits starts with a backslash, so leaving an
    input backslash alone let a caller send the literal text `\\x2c` and receive
    it back identical to a comma the server had escaped. The header's whole
    justification is telling a caller what it actually asked for, which it
    cannot do if two different requests produce the same answer.
    """
    from elvenspeak.api import _ascii_safe

    assert _ascii_safe("a, b") == "a\\x2c b"
    # The literal text, sent by a caller, must not collide with the above.
    assert _ascii_safe("a\\x2c b") != _ascii_safe("a, b")
    assert _ascii_safe("a\\x2c b") == "a\\x5cx2c b"
    # Still one shape for every escape, rather than unicode_escape's short forms.
    assert _ascii_safe("\r\n") == "\\x0d\\x0a"


def test_a_voice_id_with_a_backslash_round_trips_unambiguously(client):
    response = client.post(
        "/v1/text-to-speech/back%5Cslash/stream", json={"text": "hello"}
    )
    assert response.status_code == 200
    assert response.headers["x-elvenspeak-voice-requested"] == "back\\x5cslash"


def test_multi_speaker_voices_are_reported_in_the_listing(client):
    """Read from the sidecar and now said out loud.

    There is no ElevenLabs field to select a speaker with, so a multi-speaker
    model always speaks as its default — better stated in the listing than
    discovered by listening.
    """
    body = client.get("/v1/voices").json()
    assert body["voices"]
    assert all("speakers" in voice["labels"] for voice in body["voices"])


@pytest.mark.parametrize("sent", ["", "   ", "-"])
def test_a_blank_language_is_not_reported_as_a_dropped_preference(client, sent):
    """The spelling a form or a JS client sends for "unset", end to end.

    `spoken_language` has always answered `None` for these in isolation, and the
    request still arrived carrying one: `SpeechRequest.requested` reads its
    fields through `model_fields` and asked only whether the value `is not None`,
    which `""` passes. So `language_code` entered the candidate list, `spoke`
    compared a real voice's language against a blank and came back false, and the
    header told a caller who had expressed no preference that theirs was dropped.

    Asserted through the endpoint rather than on the validator, because the two
    halves — what counts as asked for, and what counts as spoken — are read in
    different modules and it was their disagreement that shipped.
    """
    response = client.post(
        f"/v1/text-to-speech/{VOICE}/stream",
        json={"text": "hello", "language_code": sent},
    )
    assert response.status_code == 200
    assert "language_code" not in response.headers.get("x-elvenspeak-ignored", "")
