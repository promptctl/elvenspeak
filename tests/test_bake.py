"""The image's voice-baking step, run by a test instead of by `docker build`.

These are the checks the `python -c` string this module replaced made
impossible. `tests/test_dockerfile.py` could parse that string and resolve its
names, which caught a function that had been deleted; it could never catch a
function that still existed, still took those arguments, and had quietly stopped
guaranteeing what the build depended on. That is precisely what shipped, and the
only fix is code a test can call — so these call it.

No network and no real model: `_install` opens the `.onnx.json` sidecar and never
the weights, so a placeholder file is a voice as far as this step can tell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DeclaredPrepared, make_voice, piper_prepared

from elvenspeak.bake import bake
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings
from elvenspeak.voices import Substitution

KEY = "en_US-lessac-medium"


def settings_for(models_dir: Path, voices: tuple[str, ...] = (KEY,)) -> Settings:
    """A Settings the bake accepts, built directly rather than through the env.

    `allow_download` is False here on purpose: it is the value the image's ENV
    carries, and every test below depends on the bake reaching `acquire`, which
    does not consult it.
    """
    return Settings(
        engine=piper_prepared(models_dir, voices=voices, allow_download=False),
        withheld=frozenset(),
        fallback=Substitution.FIRST_OFFERED,
        api_key=None,
        host="0.0.0.0",
        port=5001,
    )


def test_the_bake_returns_the_voices_it_proved_readable(tmp_path):
    """The guarantee, as a value: a described voice cannot exist unbaked.

    Asserted on the return rather than on the files, because "both files are on
    disk" is the weaker claim that let a corrupt sidecar bake a green image.
    """
    make_voice(tmp_path)

    (voice,) = bake(settings_for(tmp_path))

    assert voice.id == KEY
    assert voice.description == "Piper lessac (en_US, medium)"


@pytest.mark.parametrize(
    ("sidecar", "reason"),
    [
        ("{not json at all", "truncated mid-write"),
        (json.dumps({"audio": {"quality": "medium"}}), "no sample rate"),
        (json.dumps({"audio": {"sample_rate": 0}}), "a sample rate of zero"),
    ],
)
def test_a_corrupt_sidecar_fails_the_bake(tmp_path, sidecar, reason):
    """The regression that escaped #4 and #5, now checkable without a build.

    Each of these is a file an interrupted download or a full disk can leave
    behind, and each one used to produce an image that built green and died at
    container start — or, for the zero rate, one that started and then raised
    ZeroDivisionError inside a timestamp request. The bake is the last moment
    that failure is cheap, so it has to happen here.

    `ValueError` covers all three: `JSONDecodeError` is one, and `_describe`
    raises the other two by name.
    """
    make_voice(tmp_path)
    (tmp_path / f"{KEY}.onnx.json").write_text(sidecar, encoding="utf-8")

    with pytest.raises(ValueError):
        bake(settings_for(tmp_path))


def test_a_missing_voice_is_fetched_even_though_the_runtime_may_not_fetch(
    tmp_path, monkeypatch
):
    """That this step reaches `acquire`, asserted by what it does.

    The image turns `PIPER_ALLOW_DOWNLOAD` off so a missing model at boot fails
    the deploy instead of silently re-downloading — which leaves the build as
    the only moment a fetch is correct. This step used to say so with a constant
    of its own overriding the setting; it says so now by calling the method that
    fetches by definition, and a later edit that reached for `open` here instead
    would refuse to bake anything into an image configured the way the image is
    configured.

    The download is stubbed on the real `piper.download_voices` rather than by
    replacing `sys.modules["piper"]`: a stub package has no `__path__`, so
    `_install`'s own import of that submodule would only resolve if some earlier
    test had already cached it — which is how this file would pass in a suite
    and fail on its own.
    """
    import piper.download_voices

    fetched: list[str] = []

    def fake_download(key: str, models_dir: Path) -> None:
        fetched.append(key)
        make_voice(models_dir, key=key)

    monkeypatch.setattr(piper.download_voices, "download_voice", fake_download)

    (voice,) = bake(settings_for(tmp_path))

    assert fetched == [KEY]
    assert voice.id == KEY


def test_an_engine_with_no_assets_bakes_nothing_and_says_so():
    """The build of an image whose engine installs nothing, which must still work.

    A remote engine has no models to fetch, so its `acquire` reports no voices —
    and the bake has to treat that as a successful build with nothing to log
    rather than as a failure to install. [LAW:no-silent-failure] cuts the other
    way here: an empty result is only honest because an engine that *does* have
    assets refuses an empty voice list while parsing, so the two never arrive at
    the same value.
    """
    settings = Settings(
        engine=DeclaredPrepared(),
        withheld=frozenset(),
        fallback=Substitution.FIRST_OFFERED,
        api_key=None,
        host="0.0.0.0",
        port=5001,
    )
    assert bake(settings) == ()


def test_the_baked_voices_are_the_ones_the_environment_names(tmp_path, clean_env):
    """[LAW:one-source-of-truth] The build reads voices the way the server does.

    This step once re-implemented `Settings.from_env`'s split-and-strip under a
    comment asserting the two agreed. Pinned end to end — the spacing a caller
    naturally writes in `--build-arg PIPER_VOICES="a, b"` is the spacing that
    parsing must survive.

    The only test here that goes through the real environment, so it is also
    the only one that has to start from an empty one.
    """
    make_voice(tmp_path, key="en_US-lessac-medium")
    make_voice(tmp_path, key="en_GB-alba-medium")
    clean_env.setenv("PIPER_VOICES", "en_US-lessac-medium, en_GB-alba-medium")
    clean_env.setenv("PIPER_MODELS_DIR", str(tmp_path))

    baked = bake(Settings.from_env(ENGINES))

    assert [voice.id for voice in baked] == [
        "en_US-lessac-medium",
        "en_GB-alba-medium",
    ]
