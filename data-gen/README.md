# data-gen

Synthetic spatial-QA dataset generator. Builds a Blender-rendered scene of 3 to 6 colored shapes on a ground plane, then emits an RGB image, a depth map, a segmentation mask, a ground-truth scene JSON, a textual scene description, and a list of templated QA pairs about the scene.

Output is a flat folder where each scene is identified by a zero-padded six-digit prefix and shares its prefix across six files.

## Install

```sh
make install
```

Installs `bpy 4.5.9`, `numpy`, `pillow`, `pyyaml`, `tifffile` into a local `.venv`. Needs Python 3.11.

On Linux you also need the X11 libs listed in the top-level README before `bpy` will import.

## Run

```sh
make run        # writes dataset/, six files per scene
make inspect    # scene count, per-regime, per-qtype tallies
```

`make run` spawns `workers` processes from `config.yaml` and partitions scene indices across them. Each worker uses `random.Random(seed + i)` so output is reproducible regardless of worker count.

The default config produces 2000 scenes, balanced across six regimes (`sparse`, `dense`, `depth_ambiguous`, `size_mixed`, `same_color_cluster`, `vertical`) and 8 question types.

## Output

Per-scene files in `dataset/` (prefix `{i:06d}`):

| File                    | Contents                                                                              |
|-------------------------|---------------------------------------------------------------------------------------|
| `..._rgb.png`           | 1024 by 1024 Cycles render                                                            |
| `..._depth.png`         | Min-max-normalized depth (uint8, lo/hi from finite pixels)                            |
| `..._segmentation.png`  | Per-object index colorized via fixed palette (id 0 = background)                      |
| `..._scene.json`        | Camera intrinsics/extrinsics, per-object world + camera-frame pose, all pair relations |
| `..._description.txt`   | Deterministic textual scene description                                               |
| `..._qa.json`           | List of `{question_id, question_type, relation_type, question, answer}`               |

## Config

All generation knobs live in `config.yaml`: scene count, worker count, render resolution, Cycles samples, camera radius/elevation, placement disk radius and retries, and the active regime list.

## Next step

Copy the dataset into the inference subproject:

```sh
cp -r dataset ../inference/dataset
```
