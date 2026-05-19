# What VLMs Need to See: Input Representations for Spatial Reasoning

Matei Gardea, Ziteng (Ender) Ji, Max Yen.

Correspondence: Matei Gardea, `matei [dot] gardea [at] berkeley [dot] edu`

A controlled study of what vision-language models need to see in order to reason about space. We render synthetic 3D scenes with ground-truth geometry, derive four paired inputs from each scene (RGB, RGB + depth, RGB + text description, RGB + depth + description), and run the same spatial QA through five frozen VLMs in all four conditions. The goal is to isolate whether spatial competence comes from the visual signal or from linguistic grounding.

## Layout

```
data-gen/    synthetic scene + QA generator (Blender / Cycles)
inference/   run frozen VLMs across modality combos, write JSONL
eval/        score responses, aggregate metrics, render plots
```

Each subproject is isolated and has its own `pyproject.toml`, `config.yaml`, `Makefile`, and `README.md`.

## Prerequisites

Python 3.11, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for inference. `data-gen` uses `bpy` (Blender as a Python package) and on Linux needs the X11 libs `libxxf86vm1 libxfixes3 libxi6 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1`.

## Pipeline

Run the three subprojects in order. The handoff between them is a plain copy of two named artifacts: `dataset/` (produced by `data-gen`) and `responses.jsonl` (produced by `inference`).

1. [`data-gen/`](data-gen/README.md) generates the synthetic dataset.
2. [`inference/`](inference/README.md) runs the VLM sweep over a copy of that dataset.
3. [`eval/`](eval/README.md) scores the responses and produces metrics and plots.

A full sweep is about 360k VLM calls (5 models, 2000 scenes, 10 questions, up to 4 modality combos), which takes around 16 hours on an NVIDIA RTX PRO 6000. To test the pipeline, set `count` to something small in `data-gen/config.yaml` and run end to end.

The generated dataset and the model responses used in the paper are mirrored here: [Google Drive](https://drive.google.com/drive/u/1/folders/1tjFe_uDZUsKY0G8HfGDlxHODX9VWc51P). Drop them into `data-gen/dataset/` and `inference/responses.jsonl` to skip ahead to eval.

## Models

| Key                 | Family     | HF model id                        |
|---------------------|------------|------------------------------------|
| `qwen3_vl_4b`       | qwen       | `Qwen/Qwen3-VL-4B-Instruct`        |
| `qwen25_vl_3b`      | qwen       | `Qwen/Qwen2.5-VL-3B-Instruct`      |
| `paligemma2_3b`     | paligemma  | `google/paligemma2-3b-mix-224`     |
| `gemma3_4b`         | gemma      | `google/gemma-3-4b-it`             |
| `cosmos_reason2_2b` | qwen       | `nvidia/Cosmos-Reason2-2B`         |

Three of the five models are gated on Hugging Face. Accept the license on each model page while logged in before running inference:

- [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) (Google Gemma license)
- [`google/paligemma2-3b-mix-224`](https://huggingface.co/google/paligemma2-3b-mix-224) (Google Gemma license)
- [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) (NVIDIA Open Model License)

PaliGemma2 only accepts one image per prompt, so it is evaluated on `rgb` and `rgb+description` only.
