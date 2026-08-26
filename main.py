"""Text-to-speech over HTTP, spoken locally instead of by a third party.

Serves the one endpoint openconv's `Synthesizer` calls —
`POST /v1/text-to-speech/{voice_id}/stream` — against a Piper voice model
running on this machine. No account can disable this: the model is a file on
disk, and the only remote dependency is fetching it once, on first start.

# Why the response is MP3

openconv's client (`crates/openconv-agent/src/tts.rs` in the openconv repo)
decodes the response as MPEG audio — that decoder is not part of this
project's scope to change, and matching it exactly is what lets openconv's
`OPENCONV_TTS_URL` point here with no code change on that side. Piper itself
produces raw PCM, so this service is also, incidentally, an MP3 encoder:
`lame` does that conversion, invoked as a subprocess per request.

# Why one voice, and why voice_id is ignored

elvenreader-server resolves ElevenLabs voice IDs against a table with a
catch-all for anything it doesn't recognise — an unrecognised ID still gets a
synthesized response, in a substitute voice, rather than an error. Happy's
callers already expect that behaviour, so this service does the simplest
thing that is consistent with it: every `voice_id` gets the one voice this
server loaded at startup. A real per-voice table is future work if more than
one Piper voice is ever wanted; nothing here forecloses it.
"""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from piper import PiperVoice
from piper.download_voices import download_voice
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piper-server")

VOICE_NAME = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
MODELS_DIR = Path(os.environ.get("PIPER_MODELS_DIR", Path(__file__).parent / "models"))


def load_voice() -> PiperVoice:
    """Loads `VOICE_NAME`, downloading it into `MODELS_DIR` first if absent.

    Runs once, at startup — a request that arrives while a voice is still
    downloading is not a case this service handles, deliberately: it would
    trade one clean failure (refuse to start) for one confusing one (the
    first caller pays an unbounded, silent delay).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{VOICE_NAME}.onnx"
    if not model_path.exists():
        logger.info("downloading voice %s into %s", VOICE_NAME, MODELS_DIR)
        download_voice(VOICE_NAME, MODELS_DIR)
    return PiperVoice.load(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.voice = load_voice()
    logger.info("serving voice %s", VOICE_NAME)
    yield


app = FastAPI(lifespan=lifespan)


def get_voice(request: Request) -> PiperVoice:
    return request.app.state.voice


class SpeechRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/text-to-speech/{voice_id}/stream")
def synthesize(voice_id: str, body: SpeechRequest, request: Request) -> Response:
    voice = get_voice(request)

    pcm = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(body.text))
    if not pcm:
        # An empty PCM buffer means Piper produced nothing to say — surfacing
        # that as a normal-looking empty MP3 would read as a working call
        # that happened to be silent, which is a fault dressed as a feature.
        raise HTTPException(status_code=502, detail="synthesis produced no audio")

    sample_rate_khz = str(voice.config.sample_rate / 1000)
    lame = subprocess.run(
        [
            "lame",
            "-r",
            "-s",
            sample_rate_khz,
            "--bitwidth",
            "16",
            "--signed",
            "--little-endian",
            "-m",
            "m",
            "--quiet",
            "-",
            "-",
        ],
        input=pcm,
        capture_output=True,
    )
    if lame.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail=f"mp3 encoding failed: {lame.stderr.decode(errors='replace')}",
        )

    return Response(content=lame.stdout, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
