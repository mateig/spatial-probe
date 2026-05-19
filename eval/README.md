# eval

Scores VLM responses against gold spatial QA, aggregates accuracy along several axes, and renders plots.

## Install

```sh
make install
```

Installs `matplotlib`, `numpy`, `pandas`, `pyyaml`, `seaborn` into a local `.venv`. Needs Python 3.11. CPU only.

## Input

Two artifacts copied in from the upstream subprojects:

```sh
cp -r ../data-gen/dataset           dataset
cp    ../inference/responses.jsonl  responses.jsonl
```

`dataset/` is the gold dataset in the same flat layout that `data-gen` produces. `responses.jsonl` is the file written by `inference`.

## Run

```sh
make run        # load, merge, score, aggregate, plot
make inspect    # print headline tables from a previous run
```

The pipeline loads `responses.jsonl`, joins it with the gold QA on `(scene_id, question_id)`, scores each row with a type-dispatched lenient text matcher (yes/no, integer counts, color + shape object labels), and writes the scored rows plus aggregate CSVs and PNG plots.

Two question types are dropped at the merge step because they don't fit a clean text-match scorer: `same_side` (the data-gen template ends with "(left or right)?" which models read as multiple choice) and `id_extremum` (gold is `object N (size color shape)` and the id is the disambiguator).

## Output

```
output/
├── scored.jsonl                # one row per (model, scene, question, modality)
├── metrics/
│   ├── overall.csv             # accuracy per model (RGB only)
│   ├── by_modality.csv
│   ├── by_qtype.csv
│   ├── by_relation.csv
│   ├── by_regime.csv
│   ├── by_n_objects.csv
│   ├── modality_qtype.csv
│   ├── modality_regime.csv
│   ├── latency.csv
│   └── yes_bias.csv
└── plots/                      # 14 PNGs, prefixed 01_ ... 14_
```

The overall accuracy plot filters to RGB only so all five models are compared on identical inputs.
