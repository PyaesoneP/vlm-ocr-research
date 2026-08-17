# Reproducibility & Environment Setup

Part of [vlm-ocr-research](../README.md). The five required environments, Blackwell (sm_120) constraints, PaddleOCR-VL Docker failure modes, the MonkeyOCR CUDA build, LocateAnything-3B setup, and the cloud baseline.

---

### Environments

Five Python environments are required because of conflicting CUDA / transformers / PaddlePaddle versions.

| Env | Type | PyTorch / CUDA | Used for |
|---|---|---|---|
| `.venv` | venv | 2.11.0+cu130 / 13.0 | Qwen3-VL, baselines, MonkeyOCR client, and HF model code when the required local cache/network is available |
| `aiml` | conda | 2.12.0+cu130 / 13.0 | Nemotron OCR v2 (CUDA toolkit must match PyTorch for the C++ extension build) |
| `florencetf` | conda | 2.11.0+cu130 / 13.0 | Florence-2 (needs transformers 4.40.0, incompatible with 5.x) |
| `.venv_paddleocr` | venv | PaddlePaddle 3.3.1 native path / Docker sm120 offline preferred | PaddleOCR-VL (native path is broken on this Blackwell setup; Docker is the measured path) |
| `.venv_locateanything` | venv | CUDA-matched PyTorch / transformers 4.57.1 | LocateAnything-3B / NVLabs Eagle Embodied text localization |

```bash
source .venv/bin/activate          # most models
conda activate aiml                # Nemotron OCR v2
conda activate florencetf          # Florence-2
source .venv_paddleocr/bin/activate # PaddleOCR-VL (native path — broken on Blackwell, see below)
source .venv_locateanything/bin/activate # LocateAnything-3B
```

Blackwell (sm_120) is unsupported by stable PyTorch; this project uses 2.11.0+cu130. transformers is pinned at 5.8.1 (needed for SmolDocling's `AutoModelForMultimodalLM`).

### PaddleOCR-VL on Blackwell (Docker)

PaddleOCR-VL uses the **PaddlePaddle native engine**, not HuggingFace transformers (`from paddleocr import PaddleOCRVL`). PyPI `paddlepaddle-gpu` lacks sm_120 support. **Docker is the only reliable path on WSL2 + Blackwell**, the native `.venv_paddleocr` (PaddlePaddle 3.3.1) hangs at `paddle.to_tensor()`.

Five distinct failure modes; the last line printed before a freeze identifies which:

| Last line printed | Cause | Fix |
|---|---|---|
| `Checking connectivity to the model hosters...` | pings HF/BOS/ModelScope, stalls on bad routes | `-e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` |
| `Fetching N files...` | ~2 GB BOS download stalls outside China; `--rm` re-downloads | named volume `-v paddlex_models:/home/paddleocr/.paddlex` |
| after `generation_config.json`, before `Latency:` | JIT kernel compile or `/dev/shm` exhaustion (Docker default 64 MB) | `--shm-size=8g` + persistent container |
| worked before, now hangs immediately | stuck GPU / zombie VRAM (WSL2 leaks between `--rm` runs) | `wsl --shutdown`; set NVIDIA Control Panel → CUDA Sysmem Fallback → *Prefer No Sysmem Fallback* |
| prints `Latency:` and results, never exits | known Paddle teardown hang | `import os; os._exit(0)` at script end |

**WSL2 gotchas:** VRAM is not freed between `docker run --rm` containers (WDDM leak), so subsequent runs spill weights into shared memory over PCIe, a ~100× slowdown that looks like a freeze. `wsl --shutdown` between runs clears it; the sysmem-fallback policy converts the silent slowdown into a fast, visible OOM. **Never `docker kill`** during active CUDA work (SIGKILL leaks VRAM via dxgkrnl), always `docker stop`.

> Import conflict (2026-06-12): importing the project's `candidates` package before `PaddleOCRVL.predict()` triggers sysmem fallback even on a clean GPU. Use a standalone script with **zero project imports** and **data-only mounts**.

```bash
# Pull once (Chinese registry, be patient):
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline

# Benchmark (standalone script, data-only mounts):
docker run --rm --gpus all --network host --shm-size=8g \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True -e PYTHONUNBUFFERED=1 \
  -v paddlex_models:/home/paddleocr/.paddlex \
  -v $PWD/benchmark/test_dataset:/data:ro \
  -v $PWD/benchmark/results:/results \
  -v $PWD/scripts/bench_paddleocr_handwritten.py:/scripts/bench.py:ro \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
  python3 -u /scripts/bench.py
```

For rapid iteration, run a persistent container (`-d ... sleep infinity`) and `docker exec` into it to keep the JIT cache warm; `docker stop` to tear down.

Reference: [PaddleOCR-VL NVIDIA Blackwell tutorial](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html).

### MonkeyOCR GPU setup

Pre-built llama.cpp binaries are CPU-only (54s/image). Building from source with CUDA gives 4.27s/image (10×).

```bash
git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && cd /tmp/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server

cd build/bin && LD_LIBRARY_PATH=. ./llama-server \
  -hf dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 \
  --mmproj-offload --image-min-tokens 1024
curl -s http://localhost:8080/health   # → {"status":"ok"}
```

`-ngl 99` offloads all layers; `--mmproj-offload` puts the vision projector on GPU (critical for encoding speed). At `-c 4096` the image tokens don't fit and output truncates.

### LocateAnything-3B setup

LocateAnything-3B (NVLabs Eagle Embodied) is evaluated as a **Stage 1 text-localization candidate**, not as a direct handwriting transcription replacement. The public task is scene text detection / grounding, so CER/WER are reported only if the model emits actual text labels in `<ref>...</ref>` spans. Primary metrics are word IoU, recall/precision, reading-order τ, latency, and VRAM.

It needs an isolated environment because the released stack requires `transformers==4.57.1`, `numpy==1.25.0`, and `Pillow==11.1.0`.

```bash
# Python 3.11 is required for the pinned numpy==1.25.0 wheel.
/home/pyaes/.pyenv/versions/3.11.14/bin/python -m venv .venv_locateanything
source .venv_locateanything/bin/activate
pip install --upgrade pip

# Install CUDA-matched PyTorch first. On this Blackwell setup, use the
# same torch/CUDA family as the other working environments.
pip install torch==2.11.0 torchvision==0.26.0

# Then install the LocateAnything stack:
pip install -r requirements-locateanything.txt

# Smoke test one cropped handwriting image:
python candidates/locateanything/eval.py benchmark/test_dataset/handwritten/a04-039.png

# CPU fallback is intentionally disabled for benchmarks. For a slow parser-only
# smoke test on CPU, opt in explicitly:
LOCATEANYTHING_ALLOW_CPU=1 python candidates/locateanything/eval.py benchmark/test_dataset/handwritten/a04-039.png

# Full word-level localization benchmark:
LOCATEANYTHING_MAX_IMAGE_SIDE=1024 LOCATEANYTHING_MAX_NEW_TOKENS=2048 \
  python scripts/eval_wordlevel_iou.py locateanything
```

Outputs:

- `benchmark/results/locateanything_wordlevel_handwritten.json`
- `benchmark/visualizations/locateanything_wordlevel/`

The model is under NVIDIA's non-commercial research license. Do not add it to the CER leaderboard unless the benchmark confirms transcription-quality labels; otherwise compare it only in the word-level localization table.

On 12 GB VRAM, full-resolution pages OOM in the vision encoder unless optimized attention is available. The candidate resizes inference images to `LOCATEANYTHING_MAX_IMAGE_SIDE` (default 1024) and maps predicted boxes back to original image coordinates before scoring. It also uses an explicit word-level prompt, then retries the broader scene-text prompt when the first pass emits fewer than `LOCATEANYTHING_MIN_WORD_BOXES` boxes (default 20). Current measured result: word IoU 0.592, CER 0.722, τ 0.176, 12.6s/image, 11.9 GB peak VRAM.

### Cloud baseline

**16.7s end-to-end** (Doc AI 2.8s + Gemini 3.5 Flash 12.4s). `gemini-3.5-flash` is a preview model available only in the `global` region on Vertex AI (`asia-southeast1` returns 404). Document AI runs in `asia-southeast1` via a regional endpoint.

```bash
gcloud auth application-default login
set -a && source .env && set +a   # GCP_PROJECT, DOCAI_PROCESSOR_ID, GEMINI_MODEL=gemini-3.5-flash
.venv/bin/python -u benchmark/baseline.py
```

`baseline.py` prefers an API key, falls back to Vertex AI via ADC (production-tier quota; free-tier keys at 20 req/day are unusable for benchmarking).

