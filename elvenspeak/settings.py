"""The one place this process reads its environment.

[LAW:parse-dont-validate] Everything downstream of [`Settings.from_env`] runs on
values already known to be well-formed, so no handler asks whether a port was a
number. A bad environment stops the process at startup, naming every problem at
once, rather than surfacing as a 500 on somebody's first call.

A bad environment includes one that names a setting this build does not have.
[`PREFIX`] is the namespace this module answers for and [`VOCABULARY`] is
everything in it that means anything here — derived from the [`Settings`] fields
themselves, so the two cannot drift and adding a setting is adding a field. A
name in that namespace outside that set was previously read by nobody and
complained about by nobody, which is not a hypothetical: a jobspec carried
`ELVENSPEAK_CONCURRENT_SYNTHESES` to an image that predated the setting, and the
allocation came up green, showed the variable in its env, and ran eight-wide.

# What is here and what is the engine's

Only what is true whichever engine is running: which engine that is, which voice
answers for an id this server does not know, which capabilities this deployment
withholds, whether a key is required, and where to listen. Where model files live
and whether they may be fetched are Piper's business and are parsed by Piper —
held here, they would be fields every other engine is handed and ignores.

The reverse mistake is the one this module made longest. Whether to offer
timestamps was spelled `ELVENSPEAK_TIMESTAMPS` and parsed by Piper, so a
deployment that switched it off and then ran a different engine got timestamps
anyway, silently: one engine's private name for a thing every engine has. It is
[`Settings.withheld`] now, parsed here once against the shared vocabulary, and
the engine an operator happens to be running cannot change the answer.

The engine's problems still arrive in this module's list, at this module's one
moment, because [`Settings.from_env`] splices them in. Separating the settings
did not separate the report.

[LAW:one-way-deps] The registry of real engines is taken as an argument rather
than imported. Reaching for it here would make `api`, which imports this module
for its API key, transitively import every engine's third-party library — and
the reusable half of this package would stop being importable without them.
"""

from __future__ import annotations

import difflib
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

from . import memory, models
from .engine import Capability
from .provisioning import ConfigError, Prepared, Registry
from .voices import Fallback, Substitution

#: The prefix this module answers for, and the one thing that decides what
#: [`Settings.from_env`] is entitled to refuse. Every other prefix in the
#: environment belongs to an engine and is parsed privately by it — `PIPER_*` at
#: [`elvenspeak.piper.configure`], `KOKORO_*`, `CHATTERBOX_*`, `ROUTER_*` — so a
#: name outside this prefix is never this module's to have an opinion about.
#:
#: The names below are built from it rather than spelled out, so a variable this
#: module reads cannot fall outside the prefix it checks. Spelled twice, they
#: could: a mistyped `ELVESPEAK_` constant would be read by `from_env` and
#: skipped by the check, which is a variable nothing can complain about — the
#: exact silence this whole epic exists to end.
PREFIX = "ELVENSPEAK_"

#: The variables this module reads, one constant each. Named rather than written
#: at the read site because a [`Settings`] field declares which one fills it, and
#: the declaration and the read must be the same object rather than two spellings
#: of one name ([LAW:one-source-of-truth]). That declaration is what makes
#: [`VOCABULARY`] derivable, and it is why no list of legal names exists anywhere
#: in this package: the fields are the list.
ENGINE = PREFIX + "ENGINE"
FALLBACK_VOICE = PREFIX + "FALLBACK_VOICE"
API_KEY = PREFIX + "API_KEY"
WITHHOLD = PREFIX + "WITHHOLD"
CONCURRENT_SYNTHESES = PREFIX + "CONCURRENT_SYNTHESES"

#: Unprefixed, and staying that way: they are what every service on this network
#: is given, not this one's own vocabulary. Declared here all the same so that
#: the fields they fill can name them like any other — a field filled from a
#: variable it does not declare is a variable `tests/conftest.py` stops clearing.
#: Being outside [`PREFIX`], they are never candidates for the refusal below,
#: which is how that rule stays scoped without an exception list to maintain.
HOST = "HOST"
PORT = "PORT"

#: What piper really cost per concurrent synthesis on the gpu node, as
#: `(concurrent, MiB)` — `MemoryStats.Usage`, which is the figure Nomad reports,
#: since cgroup v2 leaves RSS at 0. The first row is idle with five voices loaded.
#:
#: Data rather than prose because an operator picks a ceiling off it, and it used
#: to be written out longhand everywhere it was needed ([LAW:one-source-of-truth]).
#: Now the field comment below points at it and [`_curve`] renders it for the
#: refusal, so README.md holds the only remaining literal copy of the figures —
#: which is why `tests/test_settings.py` holds that copy equal to this one, in
#: both directions.
#:
#: The last row is the one that matters: 8 concurrent is what the 4-core node's
#: CPU-derived default produced, and it died against a 2048 MiB limit.
MEASURED_PIPER = ((0, 1140), (2, 1164), (4, 1335), (8, 1773))


def _default_concurrency() -> int:
    """The bound `asyncio.to_thread` already imposes, read rather than restated.

    [LAW:one-source-of-truth] CPython computes its default executor's width as
    `min(32, (os.process_cpu_count() or 1) + 4)`, and that is the ceiling every
    synthesis in this process runs under today. Spelling the same formula here
    keeps the named default equal to the unnamed one it replaces, so turning this
    setting on changes nothing until an operator changes the number.

    PROCESS cpu count, not `os.cpu_count`, because 3.13 moved
    `ThreadPoolExecutor` onto it and this default is worth nothing if it is not
    the number the executor actually uses.

    AND IT DOES NOT MEAN WHAT IT SOUNDS LIKE. `os.process_cpu_count` follows CPU
    AFFINITY — `sched_getaffinity` on Linux, plus the `PYTHON_CPU_COUNT`
    override — and CPython does not read cgroup `cpu.max` at all. A CPU QUOTA
    does not lower it: `docker --cpus=2` on a 32-core host still answers 32, and
    so does a Nomad `resources.cpu`. Only pinning does — `--cpuset-cpus`, or
    Nomad `resources.cores`.

    Which is the opposite of reassuring, and is why it is written down. This
    default does NOT shrink to fit a quota-limited container, so on the
    deployment this feature exists for it stays as wide as the host and the
    operator has to choose the number. An earlier version of this docstring
    called the count "cgroup-aware" and said it diverges under a quota; a reader
    who believed that would conclude the ceiling self-adjusts and skip the
    setting, which is exactly the eight-way burst that OOM-killed
    elvenspeak-piper.

    `tests/test_settings.py` measures the executor's real width rather than
    checking this against a second copy of the formula — a copy cannot detect its
    own drift, which is how the `os.cpu_count` spelling survived here.
    """
    return min(32, (os.process_cpu_count() or 1) + 4)


#: Settings that named a capability in one engine's dialect, and what replaced
#: them. Refused rather than ignored: an operator who set one meant to change an
#: answer, and a name silently no longer read gives them the opposite of what
#: they asked for with nothing to notice. Data rather than a branch, so retiring
#: the next name is a line here ([LAW:dataflow-not-control-flow]).
_RETIRED = {
    "ELVENSPEAK_TIMESTAMPS": (
        f"set {WITHHOLD}=timestamps to switch them off, or unset this"
    )
}


@dataclass(frozen=True)
class Settings:
    """Everything the server needs, with no absent values left in it."""

    #: The chosen engine, configured and checked but not yet built. `main.py`
    #: opens it; the image's bake step acquires its assets. Both read this one
    #: value, which is what stops the build and the boot naming different
    #: engines.
    engine: Prepared = field(metadata={"env": ENGINE})
    #: What that engine is called: its key in the registry [`from_env`] was
    #: handed. The name used to be computed here and thrown away, which left the
    #: server able to run an engine it could not name — and the alias
    #: declarations are a file named after the engine, so a server that cannot
    #: say which engine it is running cannot find them
    #: ([`elvenspeak.voices.load_aliases`]).
    #:
    #: [LAW:one-source-of-truth] Filled from the same lookup that produced
    #: [`engine`], so the two cannot name different engines.
    engine_name: str
    #: Every engine this build could have run: the keys of the registry
    #: [`from_env`] was handed, [`engine_name`] among them.
    #:
    #: Carried because a `model_id` naming an engine this deployment is *not*
    #: running has to be refused rather than served by the one that is, and
    #: telling that apart from an id naming no engine at all is the one question
    #: neither the running engine nor the request can answer
    #: ([`elvenspeak.models.Reach`]). Names only — where those engines are and
    #: what voices they have stays theirs to advertise.
    known_engines: frozenset[str]
    #: Capabilities this deployment does not offer, whatever the engine can do.
    #: Subtracted from the engine's own declaration in
    #: [`elvenspeak.api.create_app`], which is the one place the two meet — and
    #: handed to the engine at [`Configure`] as well, so an engine that can spare
    #: itself the machinery does. Empty is the ordinary case: nothing withheld.
    withheld: frozenset[Capability] = field(metadata={"env": WITHHOLD})
    #: Which voice answers for an id this server does not know. A name, or one of
    #: [`Substitution`]'s two answers for callers who named neither a voice nor
    #: nothing. Switching substitution off makes unknown ids 404 — correct for a
    #: closed deployment, wrong for anything replacing ElevenLabs, which is why
    #: it is not the default.
    fallback: Fallback = field(metadata={"env": FALLBACK_VOICE})
    #: The value callers must present in `xi-api-key`. `None` accepts every
    #: request, which is the right default for a service on a private network
    #: and the wrong one anywhere else.
    api_key: str | None = field(metadata={"env": API_KEY})
    host: str = field(metadata={"env": HOST})
    port: int = field(metadata={"env": PORT})
    #: How much synthesis this process does at once. Callers beyond it wait
    #: rather than being refused, so a burst is slower than the limit rather than
    #: fatal to it.
    #:
    #: Bounds synthesis WORK, not requests in flight. On the streaming endpoints
    #: a permit is held while a chunk is being made and released while the caller
    #: reads: between chunks that request is blocked behind its own socket, and
    #: charging it a slot there would bill a slow reader against a budget
    #: measured for models doing work. The number of open streams stays
    #: unbounded, as it was.
    #:
    #: Nothing special-cases the router. `Remote.speak` is an ordinary engine, so
    #: a router deployment gates its own reads from backends by this same number
    #: — which is worth setting there for the fan-out that process needs rather
    #: than for a model's memory, since it synthesises nothing itself.
    #:
    #: THIS NUMBER ALREADY EXISTED; it was just nobody's. Every synthesis
    #: dispatches through `asyncio.to_thread`, whose default executor is exactly
    #: [`_default_concurrency`] wide -- so the ceiling on concurrent
    #: synthesis, and therefore on memory, was a function of the node's CORE
    #: COUNT. On the 4-core gpu node that is 8, and an eight-way burst is what
    #: OOM-killed elvenspeak-piper at a 2048 MiB limit (piper-memory-9rc). The
    #: default below is that same number, named: nothing changes behaviour until
    #: a deployment chooses, and choosing is now possible. That equality is why
    #: the bound is over work rather than over requests — a per-request bound
    #: would have capped in-flight streams, which nothing capped before, making a
    #: real capacity change out of a default nobody set.
    #:
    #: [`MEASURED_PIPER`] is what it really cost on that node -- pick from that
    #: rather than from taste. The figures are not written out here: a table
    #: copied to each place that needs it is one chance per copy to update all
    #: but one of them.
    #:
    #: A CPU-derived default for a memory bound is the wrong unit on purpose: it
    #: preserves today's behaviour exactly. The right unit is the deployment's own
    #: memory limit — which this process CAN know, and reads
    #: ([`elvenspeak.memory`]). It still does not pick a number from it: how much
    #: a synthesis costs is engine-, voice- and text-dependent, so arithmetic over
    #: that limit would look principled and be a guess. What it does instead is
    #: refuse to start when it is confined and nobody chose, below.
    #:
    #: This comment used to end "which this process cannot know. Set it where that
    #: limit is set." That was false, and it was load-bearing: it is the sentence
    #: that justified leaving the number to a jobspec nobody edited, through four
    #: OOM kills.
    #: [LAW:one-source-of-truth] The default is the same callable [`from_env`]
    #: uses, not a second spelling of the formula: a constructed `Settings` and a
    #: parsed one must not disagree about what "unset" means.
    concurrent_syntheses: int = field(
        default_factory=_default_concurrency, metadata={"env": CONCURRENT_SYNTHESES}
    )
    #: Whether a deployment chose [`concurrent_syntheses`], as against inheriting
    #: it from the host's core count.
    #:
    #: Provenance, not a second copy of the value, and it cannot be recovered from
    #: the number: an operator is perfectly entitled to choose exactly what the
    #: default would have been, and [`unsized`] must not mistake that deployment
    #: for one that chose nothing. Defaults to False so that a `Settings` built in
    #: code — every test that constructs one directly — is treated as having
    #: chosen nothing, which is the conservative reading.
    concurrency_chosen: bool = False

    @staticmethod
    def from_env(
        engines: Registry, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        env = os.environ if environ is None else environ
        problems: list[str] = []

        # Stripped like the values it is compared against downstream. A trailing
        # space in a .env file once made a plainly-present voice report missing.
        fallback_text = env.get(FALLBACK_VOICE)
        fallback: Fallback = (
            Substitution.FIRST_OFFERED
            if fallback_text is None
            else (fallback_text.strip() or Substitution.OFF)
        )

        port_text = env.get(PORT, "5001")
        try:
            port = int(port_text)
        except ValueError:
            problems.append(f"{PORT}={port_text!r} is not a number")
            port = 0
        else:
            # Parsing is not validating: -1 and 99999 are integers and neither is
            # a port. Caught here so it joins the list this module exists to
            # produce, instead of failing later inside uvicorn with a worse
            # message.
            if not 1 <= port <= 65535:
                problems.append(f"{PORT}={port} is outside 1-65535")

        # Parsed like PORT and for the same reason: an operator with two bad
        # numbers should read both on the first run, not one per restart.
        # Stripped, and empty read as unset, like every other optional variable
        # in this block: `ELVENSPEAK_API_KEY` via `or None`, the fallback voice
        # via `.strip() or ...`, the withheld list by filtering on truthiness.
        # The README documents this one as `NAME=` with nothing after it, and a
        # Nomad `env { NAME = "" }` is the same shape, so treating empty as a
        # malformed number would crashloop a service on the documented spelling.
        concurrency_text = (env.get(CONCURRENT_SYNTHESES) or "").strip()
        concurrent_syntheses = _default_concurrency()
        concurrency_chosen = bool(concurrency_text)
        if concurrency_text:
            try:
                concurrent_syntheses = int(concurrency_text)
            except ValueError:
                problems.append(
                    f"{CONCURRENT_SYNTHESES}={concurrency_text!r} is not a number"
                )
            else:
                # Zero is refused rather than read as "no limit". A gate that is
                # sometimes absent is a second mode nothing else here has, and the
                # deployment that wants no practical bound can say so with a
                # number ([LAW:no-mode-explosion]).
                if concurrent_syntheses < 1:
                    problems.append(
                        f"{CONCURRENT_SYNTHESES}={concurrent_syntheses} is not at least 1"
                    )

        problems += [
            f"{name} is no longer read; {advice}"
            for name, advice in _RETIRED.items()
            if name in env
        ]

        # [LAW:no-silent-failure] A name in this module's prefix that no field is
        # filled from is a setting an operator believes they made: it is visible
        # in the allocation's env, the service comes up green, and the process
        # runs on the value they meant to change. That is how a jobspec came to
        # carry `ELVENSPEAK_CONCURRENT_SYNTHESES` against an image that predated
        # it, eight-wide, until the OOM kill.
        #
        # Joined to `problems` rather than raised, so that an operator with a
        # misspelling and a bad port reads both on the first run — the promise
        # this module's docstring makes and, for this class of fault, did not
        # keep.
        problems += [
            f"{name} is not read by this build; it reads {_nearest(name)}"
            for name in sorted(env)
            if name.startswith(PREFIX) and name not in VOCABULARY
        ]

        try:
            withheld = _withheld(env)
        except ValueError as error:
            problems.append(str(error))
            # Carried on with, rather than raised on, so that the engine's own
            # complaints still reach the list below. An operator with a typo here
            # and a bad voice name should read both on the first run.
            withheld = frozenset()

        try:
            chosen = _prepare(engines, env, withheld)
        except ConfigError as error:
            # Spliced rather than replacing: a bad port and a bad voice name are
            # both true at once, and an operator should see both on the first
            # run rather than one per restart.
            raise ConfigError(problems + error.problems) from None

        if problems:
            raise ConfigError(problems)

        return Settings(
            engine=chosen.engine,
            engine_name=chosen.name,
            known_engines=frozenset(engines),
            withheld=withheld,
            fallback=fallback,
            api_key=env.get(API_KEY) or None,
            host=env.get(HOST, "0.0.0.0"),
            port=port,
            concurrent_syntheses=concurrent_syntheses,
            concurrency_chosen=concurrency_chosen,
        )


def _declared_by(settings: type) -> frozenset[str]:
    """The environment variables a settings dataclass declares it is filled from.

    [LAW:one-source-of-truth] The whole of this module's vocabulary, read off the
    fields that consume it. There is no list to keep beside the dataclass, so
    there is nothing to forget to update: a field that declares no variable is
    one nothing sets — [`Settings.engine_name`], [`Settings.known_engines`] and
    [`Settings.concurrency_chosen`] are all computed from other answers — and a
    field that declares one is legal by having said so.

    Takes the class rather than reading [`Settings`] out of the module, so that
    the derivation can be shown to be by-field on a dataclass it has never seen
    ([LAW:behavior-not-structure]): a test that only ever asks it about
    `Settings` cannot tell a derivation from a hard-coded answer.
    """
    return frozenset(
        item.metadata["env"] for item in fields(settings) if "env" in item.metadata
    )


#: Every variable [`Settings.from_env`] is filled from, this module's own and the
#: two unprefixed ones. Derived, so adding a setting is adding a field.
READS: frozenset[str] = _declared_by(Settings)

#: Every variable a startup here answers for by name: [`READS`], plus the
#: [`_RETIRED`] names that are refused rather than ignored. The wider of the two
#: because an operator who exports a retired name has still named something this
#: build knows about, and telling them it is unread would be worse than useless —
#: the retirement notice says what replaced it.
#:
#: This is the set a name in [`PREFIX`] is measured against, and it is what
#: `tests/conftest.py` clears between tests, rather than either place keeping a
#: list of its own.
VOCABULARY: frozenset[str] = READS | frozenset(_RETIRED)


def _nearest(name: str) -> str:
    """[`READS`], ordered by resemblance to `name`, nearest first.

    The whole set every time rather than a suggestion when one is close enough.
    "Did you mean X?" has two shapes — one with a suggestion and one without —
    and the second is what an operator who invented a name out of whole cloth
    gets, which is exactly the operator for whom the real names are the answer
    ([LAW:dataflow-not-control-flow]). One shape, sorted, always.

    [`_RETIRED`] names are absent on purpose: they are answered for, not read,
    and offering one as the near miss would send an operator to a variable that
    stops the boot on its own terms.
    """
    return ", ".join(
        difflib.get_close_matches(name, sorted(READS), n=len(READS), cutoff=0.0)
    )


def _curve() -> str:
    """[`MEASURED_PIPER`] as one line an operator can pick a number off."""
    return ", ".join(
        f"{'idle' if at == 0 else f'{at} concurrent'} {mib}" for at, mib in MEASURED_PIPER
    )


def unsized(
    settings: "Settings", confinement: memory.Limit | None = None
) -> str | None:
    """Why this deployment must not serve, when that is the case.

    One question: is this process confined to a memory limit while its synthesis
    ceiling was left to the host's core count? That pairing is the configuration
    that OOM-killed elvenspeak-piper four times, and it is indistinguishable from
    a working one until the burst arrives — the default is derived from CORES, so
    on the 4-core gpu node it was 8, and 8 is what died against 2048 MiB.

    REFUSES RATHER THAN CHOOSING. Picking a ceiling from the limit would be
    arithmetic over a per-synthesis cost that varies by engine, voice and text:
    principled-looking, and invented. Clamping is the same guess one step
    quieter — it boots, it works, and the operator's intent was overruled in a log
    line nobody tails. This project refuses rather than guesses everywhere else it
    meets this shape: `ROUTER_CONSUL_URL` is never defaulted, an unrecognised
    withheld capability is refused rather than skipped, and the engines will not
    hunt for a working espeak library.

    NARROW ON PURPOSE. An unconfined process is untouched, which is every
    development machine and every `docker run` without `--memory`. Only a
    deployment that set a memory limit and then left the width to the host's core
    count is refused.

    Called from the composition root rather than from [`Settings.from_env`],
    because the parse is shared with `python -m elvenspeak.bake` and that step
    synthesizes nothing. A refusal on the shared parse would fail image builds
    under a memory-limited builder — and CI now runs each image before it is
    pushed (piper-build-b4h.2) under a memory limit WITH a ceiling chosen
    (piper-build-b4h.scw), so a refusal that fires only when none was chosen is
    still not something any check here would see. The first evidence would be a
    publish that spent a dated tag on an image that cannot boot.
    """
    limit = memory.limit() if confinement is None else confinement
    if settings.concurrency_chosen or not isinstance(limit, int):
        return None
    return (
        f"{CONCURRENT_SYNTHESES} is unset and this process is limited to "
        f"{limit // 1048576} MiB. The default is derived from CPU cores "
        f"({settings.concurrent_syntheses} here), not from memory, so it does not "
        f"shrink to fit that limit -- an eight-way burst against 2048 MiB is what "
        f"OOM-killed elvenspeak-piper. Choose from measured cost -- piper on that "
        f"node, MiB resident: {_curve()}"
    )


@contextmanager
def reported_or_exit() -> Iterator[None]:
    """Runs a startup, turning any configuration problem into a clean exit 2.

    [LAW:single-enforcer] Every entry point comes through here — `uv run
    main.py`, the factory behind `uvicorn main:build --factory`, and the image's
    `python -m elvenspeak.bake` step — so a misconfiguration is reported one way
    whichever one is running. This was `main.py`'s private helper, which is the
    shape that lets the next entry point answer a bad environment with a raw
    traceback: the divergence gets written by omission, in the module that never
    knew the helper existed.

    A block rather than a wrapper around the parse, because not every
    configuration problem is discoverable while parsing. Whether the fallback
    voice names one the engine offers cannot be answered until the engine is
    open, so [`Catalog`] is where that check lives — and a reporter that spanned
    only [`Settings.from_env`] left exactly that problem coming out as an
    unhandled traceback, which is the one failure mode this whole module exists
    to prevent. The span is the startup, not the parse.
    """
    try:
        yield
    except ConfigError as error:
        # Every problem at once, on stderr, with a non-zero exit: an operator
        # bringing this up for the first time should not discover their
        # configuration one restart at a time.
        for problem in error.problems:
            print(f"config error: {problem}", file=sys.stderr)
        raise SystemExit(2) from None


def _withheld(env: Mapping[str, str]) -> frozenset[Capability]:
    """The capabilities this deployment switched off, or a complaint about the names.

    [LAW:one-source-of-truth] Spelled against [`Capability`], which is already the
    closed vocabulary the server and every engine share, so "what can be switched
    off" has one definition and this setting cannot name something no endpoint
    gates. The member name in any case is the spelling, which is also what the
    startup log prints — an operator reads back what they wrote.

    [LAW:no-silent-failure] An unrecognised name is refused rather than skipped
    over. A typo that withheld nothing would answer with the timestamps an
    operator had asked the service not to give, which is exactly the silent
    disagreement this setting exists to end, arriving one spelling later.

    Naming a capability the running engine never had is not an error and never
    becomes one: what this returns is subtracted from a set, and subtracting
    something absent is how sets already behave.
    """
    named = [text.strip() for text in env.get(WITHHOLD, "").split(",") if text.strip()]
    unknown = [text for text in named if text.upper() not in Capability.__members__]
    if unknown:
        raise ValueError(
            f"{WITHHOLD} names {', '.join(repr(text) for text in unknown)}, "
            f"which is not a capability; choose from "
            f"{', '.join(item.name.lower() for item in Capability)}"
        )
    return frozenset(Capability[text.upper()] for text in named)


@dataclass(frozen=True)
class _Chosen:
    """One engine, and the name it was chosen by.

    [LAW:parse-dont-validate] The type [`_prepare`] returns, and it carries both
    halves because both are results of the one lookup that proved the name real.
    Returned separately — or recomputed by a caller reading `ELVENSPEAK_ENGINE`
    a second time — they would be two answers to "which engine is this" with
    nothing holding them together.
    """

    name: str
    engine: Prepared


def _prepare(
    engines: Registry, env: Mapping[str, str], withheld: frozenset[Capability]
) -> _Chosen:
    """The named engine, configured from `env`, or a [`ConfigError`] saying why not.

    [LAW:parse-dont-validate] Its own unit, returning a type that could not exist
    before the name was checked, and failing loudly rather than returning an
    unconfigured stand-in. Nothing downstream re-asks whether the engine is real,
    because nothing downstream holds a name.

    An unnamed engine is the registry's first entry rather than a literal here.
    A default spelled in this module is a second answer to "which engines exist",
    free to name one that does not ([LAW:one-source-of-truth]).

    Named-but-blank is not unnamed. `ELVENSPEAK_ENGINE=` is a present key, which
    an operator reaches by interpolating an unset variable into a compose file —
    and taking it for "no preference" would boot the default engine, silently,
    for someone whose whole intent was to run a different one.
    """
    if not engines:
        # Before `next(iter(engines))`, which would otherwise be a bare
        # StopIteration — the one way out of this module that is not a
        # ConfigError, and so the one `reported_or_exit` cannot turn into a
        # clean exit. Nothing in this repository registers an empty one, but
        # `Registry` is a plain mapping a caller supplies and the type cannot
        # say it is non-empty.
        raise ConfigError(["no engines registered"])

    named = env.get(ENGINE, "").strip()
    if ENGINE in env and not named:
        raise ConfigError([f"{ENGINE} is empty; name an engine or unset it"])
    chosen = named or next(iter(engines))
    configure = engines.get(chosen)
    if configure is None:
        raise ConfigError([f"{ENGINE}={chosen!r} is not one of: {', '.join(engines)}"])
    # [LAW:one-source-of-truth] The one place the chosen name and the registry's
    # keys are both in hand, so it is where the engine's declared model ids are
    # read. An engine module cannot do it: it is registered under a name it has
    # never been told, which is the gap `piper-routing-7e2.17` closed by handing
    # the answer down instead of having each engine guess at its own key.
    return _Chosen(
        name=chosen,
        engine=configure(env, withheld, models.declared_by(chosen, engines)),
    )
