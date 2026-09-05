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

# No real engine here is deterministic

Piper is a VITS model sampling from a noise distribution, and the same sentence
three times gave 36352, 37888 and 37376 samples — a spread of about 4%.
Chatterbox is an autoregressive sampler and is noisier still: six syntheses of
[`LONG`] in one voice ranged 158400 to 177600 samples, a spread of about 12%.
Every comparison below between two syntheses is therefore made across a margin
that dwarfs it: a text ten times longer, a speed four times apart. A property
that needed the two calls to agree exactly would be tested against one buffer
instead, and there is no such property here.

That margin must be *declared*, not merely intended. A bare `a < b` between two
syntheses reads like a comparison across a margin and is none: for an engine
whose speed is neutralised it is the same request twice, and which call comes
back longer is the sampler's coin flip. See [`_PACE_CHANGED`], which is the
first place that mattered.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

import pytest
from conftest import (
    declaring,
    chatterbox_prepared,
    DECLARED_VOICES,
    INSTALLED_VOICE,
    MODELS_DIR,
    DeclaredEngine,
    kokoro_prepared,
    piper_prepared,
)
from fleet import cluster

from elvenspeak import router
from elvenspeak.engine import Capability, Engine, Prosody, Voice

#: One short utterance and one about ten times longer. Both are real sentences
#: because a phonemizer is entitled to make nothing of a string of `x`.
SHORT = "Hi."
LONG = (
    "Compatibility is measurable, and the measurement is this: a client written "
    "against somebody else's API reaches this one by changing a base URL."
)

#: How much shorter the fast synthesis must be before the pace counts as
#: changed, as a fraction of the slow one. A measurement, not a taste: an engine
#: that ignores speed returned fast/slow ratios of 0.892 to 1.023 across six
#: syntheses, and one that honours a 4x change returns about 0.25. This sits a
#: factor of three from each, so neither sampling noise nor a real speed change
#: lands near it.
_PACE_CHANGED = 0.7


#: [LAW:composability] The real engines fetch what they need by opening with
#: downloading on, rather than the `subject` fixture depending on an
#: asset-installing one. A fixture would have gated every parametrization on
#: every engine's assets — so `declares-everything` and `declares-nothing`, which
#: need no model, no network and no espeak-ng, would have waited on ~200 MB of
#: downloads to make a noise in memory. Provisioning is something an engine that
#: has assets does for itself, which is also what `Prepared.open` means.
def piper_engine() -> AbstractContextManager[Engine]:
    """The real thing, opened exactly as `main.build` opens it.

    Timings on, because a subject that declined the capability would take the
    conformance suite's most interesting property with it — and what an operator
    can turn off is not what the interface is being tested about.
    """
    return nullcontext(
        piper_prepared(
            MODELS_DIR, voices=(INSTALLED_VOICE,), timings=True, allow_download=True
        ).open()
    )


def kokoro_engine() -> AbstractContextManager[Engine]:
    """The second real engine, opened exactly as `main.build` opens it.

    The default export, which reports durations. The export that does not is the
    subject of `test_kokoro.py`, because what it demonstrates is a capability
    being withheld — and this suite is about engines living up to what they
    declared, whichever set that is.
    """
    return nullcontext(kokoro_prepared(MODELS_DIR, allow_download=True).open())


def chatterbox_engine() -> AbstractContextManager[Engine]:
    """The third real engine, and by a wide margin the most expensive subject here.

    It is registered anyway, and the expense is the argument for it rather than
    against it: every other subject either makes its noise in memory or opens an
    ONNX session under 150 MB, so until this one arrived the suite had never held
    an engine to the contract that could not answer instantly. What it adds is a
    0.5B autoregressive model whose audio takes longer to make than to play, and
    the properties below — a stable voice list, whole samples, a longer text
    making more audio — are exactly the ones a slow engine is tempted to fake.

    What it costs, measured: ~3.06 GiB of checkpoints fetched once, a 4.69 GiB
    resident load with a matching 4.69 GiB peak, and synthesis at 8-33x real
    time on `cpu` — so the handful of utterances below are minutes rather than
    seconds.
    `conftest.CHATTERBOX_DEVICE` is how a machine with an accelerator says so and
    gets the same tests several times faster.

    One speaker and one language, so this subject offers a single voice. The
    voice *product* — that `<speaker>-<language>` is the id and that two ids can
    be the same person — is this engine's own property rather than the seam's,
    and `test_chatterbox.py` asserts it from descriptions instead of by
    synthesizing a second voice nothing here would ask a different question of.
    """
    return nullcontext(
        chatterbox_prepared(MODELS_DIR, allow_download=True).open()
    )


@contextmanager
def router_engine() -> Iterator[Engine]:
    """The router over a real elvenspeak server, opened as `main.build` opens it.

    The fourth real engine, and the one whose conformance is least obvious: every
    property below has to survive a round trip through HTTP, an encode to PCM and
    a decode back, and — for the timed path — an alignment that was built from
    the backend's durations and has to be turned back into durations here.

    Its backend is the `DeclaredEngine` stand-in rather than Piper, because what
    is under test is the *routing* keeping the contract, and a real model behind
    it would only make the same assertions slower. That the fleet is real HTTP is
    the part that matters, and it is real.
    """
    with cluster(("declared", DECLARED_VOICES, frozenset(Capability))) as consul:
        yield router.configure({router.CONSUL_URL: consul}, frozenset(), frozenset({"router"})).open()


#: Every engine this project can put behind the API surface, and the suite below
#: is what each of them has to pass. A new engine is a line here.
#:
#: The two declared engines are the same class twice, differing only in the value
#: they were built with, which is the interface's own argument for capabilities
#: as data made into a test fixture. `frozenset(Capability)` rather than the
#: members spelled out, so a capability added to the enum is immediately claimed
#: by a subject that then has to live up to it.
#:
#: Every entry hands back a context manager, because one of them has a lifetime:
#: the router's engines are other servers, and they have to be running for as
#: long as it is asked anything. An engine with nothing to tear down says so with
#: `nullcontext` rather than by being a different shape, so `subject` builds all
#: of them one way ([LAW:dataflow-not-control-flow]).
ENGINES = [
    pytest.param(
        lambda: nullcontext(DeclaredEngine(declaring(frozenset(Capability)))),
        id="declares-everything",
    ),
    pytest.param(
        lambda: nullcontext(DeclaredEngine(declaring(frozenset()))), id="declares-nothing"
    ),
    # An engine offering a measured voice beside an unmeasured one, which is the
    # only subject on which "reads the claim it was handed" can fail. Every other
    # entry stamps one set over all its voices, so an implementation consulting
    # its own state instead of the voice passes them all — which is exactly the
    # bug this branch introduced into `DeclaredEngine` and had to fix.
    pytest.param(
        lambda: nullcontext(
            DeclaredEngine(
                (
                    *declaring(frozenset(Capability), DECLARED_VOICES[:1]),
                    *DECLARED_VOICES[1:],
                )
            )
        ),
        id="voices-differ",
    ),
    pytest.param(piper_engine, id="piper"),
    pytest.param(kokoro_engine, id="kokoro"),
    pytest.param(chatterbox_engine, id="chatterbox"),
    pytest.param(router_engine, id="router"),
]


@pytest.fixture(scope="module", params=ENGINES)
def subject(request) -> Iterator[Engine]:
    """One engine under test, built once for the whole module.

    Building is the expensive part for a real one — Piper opens a 60 MB ONNX
    session, Kokoro a 114 MB one — and it is also the part the interface
    promises happens before anything is served, so paying it once here is
    faithful as well as cheap.

    Entered rather than merely called, because the router's engines are other
    processes and stopping them is part of building it. Every subject is entered
    the same way whether or not it has anything to tear down.
    """
    with request.param() as engine:
        yield engine


@pytest.fixture
def measuring(subject: Engine) -> tuple[Engine, tuple[Voice, ...]]:
    """`subject` and every distinct claim of its that measures utterances.

    The capability check lives in a fixture rather than at the top of each test
    so that the tests themselves are unconditional: what a test asserts should
    not be entangled with whether it applies. It is also the property from the
    other side — a voice that did not declare this is one the server never asks
    about, so a suite that asked anyway would be testing a call that cannot
    happen.
    """
    return subject, speaking_with(subject, Capability.TIMESTAMPS)


def one_per_claim(subject: Engine) -> tuple[Voice, ...]:
    """One voice for each distinct capability set this engine offers.

    The properties below are about what an engine does with the claim it was
    handed, so two voices making the same claim exercise one path twice. An
    engine whose voices are alike therefore costs exactly what a single voice
    cost before — kokoro's pace check alone is fifteen seconds of real synthesis —
    while one whose voices differ is measured on every claim it makes, which is
    the only place the property is falsifiable at all.

    [LAW:dataflow-not-control-flow] The data decides how many syntheses happen,
    not a branch on which engine this is.
    """
    seen: dict[frozenset[Capability], Voice] = {}
    for voice in subject.voices():
        seen.setdefault(voice.capabilities, voice)
    return tuple(seen.values())


def speaking_with(subject: Engine, capability: Capability) -> tuple[Voice, ...]:
    """Every distinct claim this engine makes that includes `capability`.

    Skips rather than passing vacuously when it makes none: an engine that did
    not declare this is one the server never asks, so a suite that asked anyway
    would be testing a call that cannot happen. Asked across all its claims and
    not only the first voice's, because an engine offering a measured voice
    beside an unmeasured one has to be held to it for the one that measures.
    """
    voices = tuple(
        voice for voice in one_per_claim(subject) if capability in voice.capabilities
    )
    if not voices:
        pytest.skip(f"no voice declares {capability.name}")
    return voices


def samples_of(subject: Engine, voice: Voice, text: str, speed: float = 1.0) -> int:
    """How much audio `subject` makes of `text`, in samples."""
    spoken = subject.speak(voice, text, Prosody(speed=speed))
    return len(b"".join(spoken.audio)) // 2


def first_voice(subject: Engine) -> Voice:
    """The voice the tests that need only one speak in, and the default voice.

    Asserted rather than skipped. An engine offering nothing is a thing
    `/health` reports rather than a crash, but no engine this project registers
    is entitled to it — and a skip here would have quietly withdrawn every test
    below from an engine whose voice list had silently emptied, which is the one
    failure they exist to catch.
    """
    voices = subject.voices()
    assert voices, "engine offers no voices"
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


def test_what_a_voice_says_it_can_do_does_not_change_between_calls(
    subject: Engine,
):
    """The negotiation happens once, at startup, and is held for the process.

    The server reads a voice's claim on the request that names it, so a claim
    that varied between calls would make the 501 gate and the ignored header two
    readings of two different facts — stale rather than wrong, which is the kind
    that never surfaces as a failure.

    Since the claim travels on a frozen `Voice`, this is now asking the same
    question as "the voice list does not change", from the one angle that matters
    for what the server promises about it.

    `models` is held to the same standard for the same reason: which engine
    answers for a voice is read on the request that names it, so a set that
    varied between calls would decide two requests differently with nothing
    having changed. Asserted non-empty as well, because equality alone is
    satisfied by a subject that stamps nothing — and a voice that names no model
    id refuses the caller who named the engine about to speak.
    """
    assert {voice.id: voice.capabilities for voice in subject.voices()} == {
        voice.id: voice.capabilities for voice in subject.voices()
    }
    assert {voice.id: voice.models for voice in subject.voices()} == {
        voice.id: voice.models for voice in subject.voices()
    }
    assert all(voice.models for voice in subject.voices())


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


def test_the_pace_varies_exactly_when_speed_is_declared(subject: Engine):
    """[`Capability.SPEED`] is honest in both directions, and both are checked here.

    This is the honesty check the ignored header cannot make. The header reports
    `voice_settings.speed` as honoured on the strength of the declaration alone;
    if the declaration is decorative, the caller is told the speed was applied
    and the audio says otherwise — rule 2 broken in the one mechanism whose whole
    job is to stop that.

    Stated as an equivalence rather than as "if declared, it varies", so the
    engine that varies its pace *without* declaring it fails too. That one is
    the more insidious of the pair: the server neutralises `speed` to 1.0 for an
    engine that did not declare it, so such an engine is never caught in
    production — it is simply an engine whose fixed rate is a fiction, and the
    day it declares the capability the header starts telling the truth about
    audio that was already varying. Unconditional for the same reason: a skip
    for the engines that declare nothing would leave exactly that case untested.
    """
    for voice in one_per_claim(subject):
        fast = samples_of(subject, voice, LONG, speed=2.0)
        slow = samples_of(subject, voice, LONG, speed=0.5)
        # [LAW:verifiable-goals] Across `_PACE_CHANGED` rather than a bare `<`,
        # so an engine that ignores speed is judged on a margin its own sampling
        # noise cannot cross.
        varies = fast < slow * _PACE_CHANGED
        assert varies == (Capability.SPEED in voice.capabilities), (
            f"{voice.id}: {fast} samples at speed 2.0 against {slow} at 0.5"
        )


def test_a_declared_measurement_accounts_for_every_sample(measuring):
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
    subject, voices = measuring
    for voice in voices:
        spoken = subject.speak_timed(voice, LONG, Prosody())

        assert spoken.sample_rate > 0, voice.id
        assert spoken.timings, voice.id
        assert sum(t.samples for t in spoken.timings) * 2 == len(spoken.pcm), voice.id
