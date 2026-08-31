"""Every property the seam promises, asked of every engine there is.

[LAW:verifiable-goals] An interface with one implementation is an untested
claim. [`elvenspeak.engine`] states its promises in prose — a stable voice list,
a constant capability set, signed 16-bit samples, timings that account for every
one of them — and until this file existed nothing checked any of them against an
implementation. An engine returning stereo audio, or a rate it does not really
speak at, or durations summing to less than the audio they describe, produces
plausible output that fails somewhere far away, in the alignment or the encoder
or a caption renderer belonging to somebody else.

The suite is parameterized over engines rather than written per engine, so a new
engine is a line in [`ENGINES`] and inherits all of it. That registry is also the
sharpest available statement of what the interface costs to satisfy: a candidate
supplies a name and a way to build one, and nothing else — no fixture, no sample
text, no metadata on the side. The day a candidate needs something extra, the
interface is short of something the server would need too.

# What this file can prove, and what it deliberately does not

"The audio is mono, at exactly the rate the result declares" is only half
checkable from here, and pretending otherwise would be worse than saying so. A
buffer of bytes does not reveal its channel count or its true rate; nothing in
this process holds an independent clock to compare against. What *is* checkable
is every consequence that shows up as arithmetic — whole samples, a positive
rate, timings summing to the sample count — and, for an engine that measures,
that last one is also the channel-count check: stereo audio doubles the samples
without changing what the phonemes lasted, and the sum stops matching.

"Streamed chunks concatenate to what the non-streamed call returns" is not
tested because the seam makes it unfalsifiable: both endpoints call the one
[`Engine.speak`], and the difference between them is whether the server joins
the chunks or forwards them. There is no second code path for an engine to
disagree with itself across. That the property collapsed into an identity is a
fact about the interface being right, and it is worth more stated here than it
would be as a test that cannot fail.

# Piper is not deterministic

It is a VITS model sampling from a noise distribution, and the same sentence
three times gave 36352, 37888 and 37376 samples — a spread of about 4%. Every
comparison below between two syntheses is therefore made across a margin that
dwarfs it: a text ten times longer, a speed four times apart. A property that
needed the two calls to agree exactly would be tested against one buffer instead,
and there is no such property here.
"""

from __future__ import annotations

import pytest
from conftest import (
    INSTALLED_VOICE,
    MODELS_DIR,
    DeclaredEngine,
    needs_installed_model,
)

from elvenspeak.engine import Capability, Engine, Prosody, Voice

#: One short utterance and one about ten times longer. Both are real sentences
#: because a phonemizer is entitled to make nothing of a string of `x`.
SHORT = "Hi."
LONG = (
    "Compatibility is measurable, and the measurement is this: a client written "
    "against somebody else's API reaches this one by changing a base URL."
)


def piper_engine() -> Engine:
    """The real thing, opened exactly as `main.build` opens it.

    Timings on, because a subject that declined the capability would take the
    conformance suite's most interesting property with it — and what an operator
    can turn off is not what the interface is being tested about.
    """
    from elvenspeak import piper

    return piper.load(
        keys=(INSTALLED_VOICE,),
        models_dir=MODELS_DIR,
        allow_download=False,
        timings=True,
    )


#: Every engine this project can put behind the API surface, and the suite below
#: is what each of them has to pass. A new engine is a line here.
#:
#: The two declared engines are the same class twice, differing only in the value
#: they were built with, which is the interface's own argument for capabilities
#: as data made into a test fixture. `frozenset(Capability)` rather than the
#: members spelled out, so a capability added to the enum is immediately claimed
#: by a subject that then has to live up to it.
ENGINES = [
    pytest.param(
        lambda: DeclaredEngine(frozenset(Capability)), id="declares-everything"
    ),
    pytest.param(lambda: DeclaredEngine(frozenset()), id="declares-nothing"),
    pytest.param(piper_engine, id="piper", marks=needs_installed_model),
]


@pytest.fixture(scope="module", params=ENGINES)
def subject(request) -> Engine:
    """One engine under test, built once for the whole module.

    Building is the expensive part for a real one — Piper opens a 60 MB ONNX
    session — and it is also the part the interface promises happens before
    anything is served, so paying it once here is faithful as well as cheap.
    """
    return request.param()


@pytest.fixture
def measuring(subject: Engine) -> Engine:
    """`subject`, but only for the engines that claim to measure utterances.

    The capability check lives in a fixture rather than at the top of each test
    so that the tests themselves are unconditional: what a test asserts should
    not be entangled with whether it applies. It is also the property from the
    other side — an engine that did not declare this is one the server never
    asks, so a suite that asked anyway would be testing a call that cannot
    happen.
    """
    return _declaring(subject, Capability.TIMESTAMPS)


@pytest.fixture
def varying(subject: Engine) -> Engine:
    """`subject`, but only for the engines that claim to vary their pace."""
    return _declaring(subject, Capability.SPEED)


def _declaring(subject: Engine, capability: Capability) -> Engine:
    if capability not in subject.capabilities():
        pytest.skip(f"engine does not declare {capability.name}")
    return subject


def samples_of(subject: Engine, voice: Voice, text: str, speed: float = 1.0) -> int:
    """How much audio `subject` makes of `text`, in samples."""
    spoken = subject.speak(voice, text, Prosody(speed=speed))
    return len(b"".join(spoken.audio)) // 2


def first_voice(subject: Engine) -> Voice:
    """The voice the tests that need only one speak in.

    An engine is entitled to offer none — `/health` reports that rather than
    failing — so this skips instead of erroring, leaving the tests that iterate
    over the enumeration to say something true about an empty one.
    """
    voices = subject.voices()
    if not voices:
        pytest.skip("engine offers no voices")
    return voices[0]


# ------------------------------------------------------- what the engine says


def test_the_voices_on_offer_do_not_change_between_calls(subject: Engine):
    """`voices()` promises a stable order, and the catalog is built from one call.

    A client reads `GET /v1/voices` and echoes an id back; the catalog that
    resolves it was built at startup from a single call to this method. An engine
    minting fresh ids, or ordering by a set's iteration, would serve a listing
    that stops matching itself between restarts with nothing here to notice.
    """
    assert subject.voices() == subject.voices()


def test_no_two_voices_share_an_id(subject: Engine):
    """A duplicate id does not collide loudly — it silently deletes a voice.

    `Catalog.for_engine` keys a dict by id, so the second voice with a given id
    replaces the first and the listing is quietly one short. Nothing downstream
    can detect that, because a catalog has no idea what it was not given.
    """
    ids = [voice.id for voice in subject.voices()]
    assert len(set(ids)) == len(ids)


def test_what_the_engine_says_it_can_do_does_not_change_between_calls(
    subject: Engine,
):
    """The negotiation happens once, at startup, and is held for the process.

    `create_app` asks exactly once, which is what makes the 501 gate and the
    ignored header two readings of one fact. An engine whose answer varied per
    call would have that fact captured at an arbitrary moment, and every later
    answer the server gave about it would be stale rather than wrong — the kind
    that never surfaces as a failure.
    """
    assert subject.capabilities() == subject.capabilities()


# --------------------------------------------------------- what it then does


def test_every_voice_on_offer_can_be_spoken_in(subject: Engine):
    """`voices()` promises "now, not eventually" — this is that promise, checked.

    A voice that would have to be fetched or warmed on first use is not supposed
    to be offered. The failure this catches is the enumeration listing something
    the engine cannot actually reach: a stale catalog entry, a model that was
    described but never opened, a remote voice that has since been withdrawn.
    """
    for voice in subject.voices():
        assert samples_of(subject, voice, SHORT) > 0


def test_audio_arrives_as_whole_samples(subject: Engine):
    """Signed 16-bit means two bytes each, and the chunking may not split one.

    The streaming encoder is pumped chunk by chunk, so a chunk carrying half a
    sample does not fail — it shifts every later sample by one byte and the rest
    of the utterance decodes as noise. An odd total is the same defect from the
    other end, and it is what stereo-in-disguise or a 24-bit engine looks like
    before anybody listens.
    """
    spoken = subject.speak(first_voice(subject), SHORT, Prosody())
    assert spoken.sample_rate > 0
    for chunk in spoken.audio:
        assert len(chunk) % 2 == 0


def test_a_longer_text_makes_more_audio(subject: Engine):
    """The one property that says the text reached the engine at all.

    An engine that ignored its argument and returned a fixed buffer — a stub
    half-wired to a real model, a cache keyed on the wrong thing — satisfies
    every other test in this file.
    """
    voice = first_voice(subject)
    assert samples_of(subject, voice, LONG) > samples_of(subject, voice, SHORT)


# ---------------------------------------------- what it claims it can also do


def test_a_declared_speed_really_changes_the_pace(varying: Engine):
    """[`Capability.SPEED`] declared, so 2.0 has to be audibly faster than 0.5.

    This is the honesty check the ignored header cannot make. The header reports
    `voice_settings.speed` as honoured on the strength of the declaration alone;
    if the declaration is decorative, the caller is told the speed was applied
    and the audio says otherwise — rule 2 broken in the one mechanism whose whole
    job is to stop that.
    """
    voice = first_voice(varying)
    assert samples_of(varying, voice, LONG, speed=2.0) < samples_of(
        varying, voice, LONG, speed=0.5
    )


def test_a_declared_measurement_accounts_for_every_sample(measuring: Engine):
    """[`TimedSpeech`]'s stated invariant, enforced nowhere until here.

    Every alignment downstream is derived under it: `align` spreads characters
    across the timings and trusts their sum to be the length of the audio. An
    engine whose durations fall short produces a timeline compressed against real
    audio — captions that drift further from the speech the longer it runs, and
    a whole class of bug reports pointing at the renderer.

    Non-empty as well as summing, because an engine returning no timings at all
    sums to zero and would pass an invariant check alone, having answered a
    capability it declared with nothing.
    """
    voice = first_voice(measuring)
    spoken = measuring.speak_timed(voice, LONG, Prosody())

    assert spoken.sample_rate > 0
    assert spoken.timings
    assert sum(timing.samples for timing in spoken.timings) * 2 == len(spoken.pcm)
