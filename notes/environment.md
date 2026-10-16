# Environment

Recorded 2026-10-16 during handoff 01 (repo setup and smoke test).

## Machine

| | |
|---|---|
| Machine | Mac mini (`Mac16,11`), shared: it also runs the gateway server |
| Chip | Apple M4 Pro |
| Memory | 48 GB |
| macOS | 26.6.2 (build 25G83) |

## Python

- Python 3.14.6 (Homebrew, `/opt/homebrew/bin/python3.14`), virtualenv in `.venv/`, pip 26.1.2.
- Rebuild: `/opt/homebrew/bin/python3.14 -m venv .venv`, then `.venv/bin/python -m pip install --no-cache-dir -r requirements.txt`.

| Package | Version |
|---|---|
| torch | 2.14.1 |
| transformers | 5.20.1 |
| accelerate | 1.15.0 |
| datasets | 5.1.0 |
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| huggingface_hub | 2.2.0 |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |
| hf-xet | 1.7.0 |

`requirements.txt` pins the first nine. The full `pip freeze` is at the end of this file.

## Pinned hashes

| What | Hash |
|---|---|
| Model revision, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| `model.safetensors` sha256 (matches the Hub's LFS hash) | `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945` |
| frictionmap commit, source of `src/before_the_wait/lexicon.py` | `6bc1552d335eab1434d93519644689f8e560e456` |

## Model files

- The Hugging Face cache lives in `.hf_cache/` (`HF_HOME`, gitignored). `scripts/smoke_test.py` sets `HF_HOME` itself.
- The model download is about 3.6 GB (3,562,046,103 bytes). `du -h` shows it as 3.3G because it counts in GiB (2^30 bytes).
- Any other code that imports `huggingface_hub`, `transformers` or `hf_xet` must set `HF_HOME` too. Otherwise files go to `~/.cache/huggingface`; importing `hf_xet` alone creates a log folder there.
- Re-download:

```bash
HF_HOME="$PWD/.hf_cache" .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', revision='ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562')"
```

## Acceptance runs

Run 2026-10-16 11:55 from the repo root.

`.venv/bin/python -m pytest -q`:

```
.................                                                        [100%]
17 passed in 0.01s
```

`.venv/bin/python scripts/smoke_test.py`, without the completion text, warnings and progress bars:

```
MPS available = True
torch 2.14.1, transformers 5.20.1
model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B @ ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
dtype torch.float16, device mps:0, layers 28, hidden size 1536
prompt: '<｜begin▁of▁sentence｜><｜User｜>What is 17 + 25?<｜Assistant｜><think>\n' (15 tokens)

29 hidden-state tensors (embeddings + 28 layers):
  [ 0] [1, 15, 1536] torch.float16
  [ 1] [1, 15, 1536] torch.float16
  [ 2] [1, 15, 1536] torch.float16
  [ 3] [1, 15, 1536] torch.float16
  [ 4] [1, 15, 1536] torch.float16
  [ 5] [1, 15, 1536] torch.float16
  [ 6] [1, 15, 1536] torch.float16
  [ 7] [1, 15, 1536] torch.float16
  [ 8] [1, 15, 1536] torch.float16
  [ 9] [1, 15, 1536] torch.float16
  [10] [1, 15, 1536] torch.float16
  [11] [1, 15, 1536] torch.float16
  [12] [1, 15, 1536] torch.float16
  [13] [1, 15, 1536] torch.float16
  [14] [1, 15, 1536] torch.float16
  [15] [1, 15, 1536] torch.float16
  [16] [1, 15, 1536] torch.float16
  [17] [1, 15, 1536] torch.float16
  [18] [1, 15, 1536] torch.float16
  [19] [1, 15, 1536] torch.float16
  [20] [1, 15, 1536] torch.float16
  [21] [1, 15, 1536] torch.float16
  [22] [1, 15, 1536] torch.float16
  [23] [1, 15, 1536] torch.float16
  [24] [1, 15, 1536] torch.float16
  [25] [1, 15, 1536] torch.float16
  [26] [1, 15, 1536] torch.float16
  [27] [1, 15, 1536] torch.float16
  [28] [1, 15, 1536] torch.float16
all finite = True, max |h| = 3896.0

generated 300 new tokens in 4.64 s = 64.6 tokens/second
```

- Hidden states: 29 tensors (embeddings + 28 layers), each `[1, 15, 1536]` fp16. All finite; max |h| = 3896 (fp16 max is 65504).
- Generation: 300 new tokens (stopped at end-of-sequence) in 4.64 s = **64.6 tokens/second**. The completion ends with `\boxed{42}`.
  - Batch 1, fp16, temperature 0.6, top-p 0.95, seed 0, max 512 new tokens.
  - Timed around `generate()` only, after the hidden-states pass had warmed up the GPU kernels.
  - Speed varies between runs: an earlier run the same day (10:45) generated the identical 300 tokens in 5.40 s = 55.5 tokens/second.
  - The gateway server was running on the same machine both times; its load wasn't checked.

## Full pip freeze

```
accelerate==1.15.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
annotated-doc==0.0.5
anyio==4.15.1
attrs==26.1.0
click==8.5.0
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
datasets==5.1.0
dill==0.4.1
filelock==4.2.0
fonttools==4.67.0
frozenlist==1.8.0
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.13.1
httpx2==2.13.1
huggingface_hub==2.2.0
idna==3.20
iniconfig==2.3.1
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
markdown-it-py==4.2.0
MarkupSafe==3.0.4
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
multidict==7.1.0
multiprocess==0.70.19
narwhals==2.27.1
networkx==3.7
numpy==2.5.4
packaging==26.3
pandas==3.0.6
pillow==12.3.0
pluggy==1.7.0
propcache==0.5.4
psutil==7.2.2
pyarrow==26.0.0
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.18.1
setuptools==84.0.0
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
threadpoolctl==3.7.0
tokenizers==0.23.3
torch==2.14.1
tqdm==4.70.1
transformers==5.20.1
truststore==0.10.5
typer==0.27.3
typing_extensions==4.16.0
xxhash==4.0.1
yarl==1.25.1
```
