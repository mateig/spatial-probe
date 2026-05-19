# inference

Runs a fixed prompt against multiple VLMs, multiple modality combos, and every QA in a `data-gen`-style dataset, and appends per-question rows to a single JSONL. Designed to be re-runnable: completed `(model, scene, question, modality)` tuples are skipped on resume.

## Install

```sh
make install
```

Installs `transformers`, `torch` (CUDA 12.8 on Linux), `accelerate`, `qwen-vl-utils`, `pillow`, `pyyaml` into a local `.venv`. Needs Python 3.11 and a CUDA GPU.

## Hugging Face token

Three of the five models are gated. Accept each license on its model page while logged in to Hugging Face:

- [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)
- [`google/paligemma2-3b-mix-224`](https://huggingface.co/google/paligemma2-3b-mix-224)
- [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)

Then create `.env` from the example and paste your token:

```sh
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...
```

## Input

This subproject expects a `dataset/` directory in the same flat layout that `data-gen` produces:

```sh
cp -r ../data-gen/dataset dataset
```

## Run

```sh
make run
```

Iterates over `models × scenes × modality combos × questions`, batches the pending questions per `(scene, modality)` chunk through a single forward pass, and appends one row per call to `responses.jsonl`. Errored rows are written with `response=null` and an `error` string so they get retried on the next run. A kill mid-run loses at most the in-flight batch.

CLI overrides on `scripts/run.py`:

| Flag                  | Effect                                                         |
|-----------------------|----------------------------------------------------------------|
| `--config <path>`     | use a different yaml                                           |
| `--models q1 q2 ...`  | only run these model keys (must be enabled in config)          |
| `--scene-ids id1 id2` | explicit scene set, overrides config selection                 |
| `--limit-scenes N`    | cap to first N (after `start`)                                 |
| `--no-resume`         | ignore `responses.jsonl`, regenerate every row                 |

## Output

One JSONL file at `responses.jsonl`. Each line:

```json
{
  "created_at": "<ISO 8601 UTC>",
  "model_key": "qwen3_vl_4b",
  "model_id": "Qwen/Qwen3-VL-4B-Instruct",
  "scene_id": "000042",
  "question_id": 3,
  "modalities": ["rgb", "description"],
  "modality_key": "rgb+description",
  "question": "...",
  "response": "yes",
  "error": null,
  "elapsed_s": 0.1234
}
```

A full sweep is roughly 360k rows.

## Next step

Copy the gold dataset and the responses into the eval subproject:

```sh
cp -r dataset           ../eval/dataset
cp    responses.jsonl   ../eval/responses.jsonl
```
