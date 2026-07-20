# Reference-Guided Hidden-View Generation for InstantMesh

_A lightweight adapter for injecting additional side and back reference details into Zero123++ multi-view generation._

> [!IMPORTANT]
> This repository is a research modification of the original [InstantMesh](https://github.com/TencentARC/InstantMesh) project. InstantMesh and its original implementation were created by the original InstantMesh authors. This project builds on their image-to-3D reconstruction pipeline and modifies the Zero123++ multi-view generation stage by adding reference-guided conditioning and spatial gating.

## Overview

[InstantMesh](https://github.com/TencentARC/InstantMesh) reconstructs a 3D object from one image. Its Zero123++ stage first predicts six views in a fixed 3×2 image sheet, which the frozen InstantMesh reconstruction model then converts into a mesh. With only a front image, details that exist solely on the side or back must be guessed; they can therefore be missing, inconsistent, or hallucinated.

This project adds explicit side/back reference images to that stage. A small trainable adapter turns frozen CLIP vision features from those images into conditioning tokens. Camera azimuth/elevation metadata determines how strongly each reference should affect each of the six target views, and a spatial mask confines that effect to the corresponding tiles of the Zero123++ sheet.

```text
Main input image
        +
Additional reference-view images
        +
Camera-view metadata
        ↓
Reference adapter
        ↓
Zero123++ with spatially gated reference conditioning
        ↓
Six-view 3×2 sheet
        ↓
InstantMesh reconstruction model
        ↓
3D mesh
```

The core contribution is **reference-guided hidden-view generation for Zero123++ using a lightweight adapter, explicit camera-view metadata, and spatial gating**. It is not an automatic retrieval system: reference images and their metadata are supplied by the user or dataset. The repository also retains a separate CLIP-based multi-seed reranking baseline, but retrieval itself is not implemented.

## Based on InstantMesh

The original image-to-3D pipeline, reconstruction model, project structure, and major supporting components come from Tencent ARC's InstantMesh. This repository does not claim ownership of the InstantMesh architecture or implementation.

The changes in this repository are focused on the reference adapter, view-aware conditioning, spatial gating, adapter-only training/checkpointing, dataset preparation utilities, ablation tools, and related experiments. Users of this work should cite the original InstantMesh paper and repository and comply with the licenses for InstantMesh, Zero123++, all pretrained models, and their datasets.

- Official repository: [TencentARC/InstantMesh](https://github.com/TencentARC/InstantMesh)
- Paper: [InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models](https://arxiv.org/abs/2404.07191)
- Authors: Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan
- Repository license: [Apache License 2.0](LICENSE)

The citation below is copied from the original InstantMesh README:

```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

## What changed

The adapter implementation is intentionally small:

1. The frozen Zero123++ CLIP vision encoder extracts one embedding per reference image.
2. `ReferenceAdapter` applies a two-layer MLP and adds a learned embedding for the reference view ID.
3. Pose metadata produces six routing weights, one for each fixed Zero123++ target pose.
4. The adapter emits one weak global token plus six routed slot tokens and appends them to the normal prompt embeddings.
5. Zero123++ runs a base branch and a reference-conditioned branch. Their difference is blended through a fixed per-tile latent mask, so reference influence is concentrated on relevant output views.

For Zero123++ v1.2, the row-major sheet layout is:

| Sheet position | Slot | Azimuth | Elevation |
|---|---:|---:|---:|
| Row 1, left | 0 | 30° | 20° |
| Row 1, right | 1 | 90° | −10° |
| Row 2, left | 2 | 150° | 20° |
| Row 2, right | 3 | 210° | −10° |
| Row 3, left | 4 | 270° | 20° |
| Row 3, right | 5 | 330° | −10° |

Spatial gating adds no trainable parameters. In adapter-only configurations, the Zero123++ UNet, VAE, vision encoder, and text encoder are frozen; `reference_unfreeze_crossattn` is disabled. The InstantMesh reconstruction network is used after view generation and is not trained by this workflow.

## Model size

The included [`training_log.txt`](training_log.txt) verifies the following model summary for the recorded experiment:

| Component | Parameters | Training state |
|---|---:|---|
| Reference adapter | 2,106,368 (reported as 2.1M) | Trainable |
| Zero123++ `RefOnlyNoisedUNet` | 865M | Frozen |
| Logged total | 868M | 2.1M trainable, 865M non-trainable |

Only the adapter state is saved by the adapter-only checkpoint callback.

## Repository map

| Path | Purpose |
|---|---|
| [`zero123plus/reference_adapter.py`](zero123plus/reference_adapter.py) | Adapter tokens, view routing, tile masks, and optional cross-attention unfreezing helper |
| [`zero123plus/model.py`](zero123plus/model.py) | Adapter-only training, frozen encoders/UNet, spatially gated prediction, validation, and metrics |
| [`zero123plus/pipeline.py`](zero123plus/pipeline.py) | Reference-aware Zero123++ inference path |
| [`zero123plus/reference_utils.py`](zero123plus/reference_utils.py) | Reference loading, metadata parsing, target poses, and slot-weight computation |
| [`run.py`](run.py) | Full Zero123++ → InstantMesh inference, including adapter and reranking modes |
| [`scripts/build_objaverse_reference_dataset.py`](scripts/build_objaverse_reference_dataset.py) | Rendering and manifest construction from local 3D assets or Objaverse-style metadata |
| [`scripts/eval_reference_adapter_ablation.py`](scripts/eval_reference_adapter_ablation.py) | Controlled Zero123++ sheet ablations and per-slot difference metrics |
| [`tests/`](tests/) | Unit tests for routing, layout, dataset tools, rendering helpers, and validation scheduling |

## Installation

The inherited InstantMesh setup targets Python 3.10, PyTorch 2.1.0, and CUDA 12.1. A CUDA-capable GPU is required by the main inference and training workflows.

```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

conda install Ninja
conda install cuda -c nvidia/label/cuda-12.1.0

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7
pip install -r requirements.txt
```

The requirements install `nvdiffrast` from NVIDIA's Git repository. Blender is additionally required to render training views from `.glb`, `.gltf`, `.obj`, or `.fbx` assets.

The scripts download `sudo-ai/zero123plus-v1.2` and InstantMesh weights when they are not already cached. You can also place the InstantMesh files referenced by the chosen inference config under `ckpts/`; for example, the low-VRAM mesh config expects `diffusion_pytorch_model.bin` and `instant_mesh_large.ckpt`.

## Reference input and metadata

Pass either one reference image or a folder of images. Explicit camera metadata is recommended because automatic view assignment is disabled. The metadata file maps each basename to numeric azimuth and elevation values:

```json
{
  "side.png": {"azimuth": 90, "elevation": -10},
  "back.png": {"azimuth": 180, "elevation": 0}
}
```

If metadata is omitted, filenames matching `ref_<azimuth>_el<elevation>` (for example, `ref_180_el0.png`) can provide a pose. Coarse labels such as `front`, `side`, or `back` can also be inferred from filenames or supplied through `--reference_view_labels`, but their routing is less precise. Unknown references receive weak routing.

### Legacy command and checkpoint compatibility

The former `--rag_*` command-line options remain accepted as deprecated aliases and emit a `FutureWarning`; new commands should use the `--reference_*` names shown below. Checkpoint loading accepts both the new `reference_adapter.*` / `model.reference_adapter.*` prefixes and the former `rag_adapter.*` / `model.rag_adapter.*` prefixes. Newly saved checkpoints use only `reference_adapter.*`.

## Inference

### Full reference-guided mesh generation

An adapter checkpoint is required for meaningful reference-guided output. No trained adapter checkpoint is committed to this repository.

```bash
python run.py configs/instant-mesh-large-lowvram.yaml path/to/main.png \
  --no_rembg \
  --enable_reference_adapter \
  --reference_images path/to/references \
  --reference_metadata path/to/ref_metadata.json \
  --reference_adapter_ckpt path/to/adapter_last.pt \
  --reference_spatial_gating \
  --reference_slot_weight_mode wide \
  --reference_cross_view_propagation_enabled \
  --save_video
```

`run.py` saves the generated six-view sheet under `outputs/<config-name>/images/`, the mesh under `outputs/<config-name>/meshes/`, and an optional turntable under `outputs/<config-name>/videos/`. Omit `--no_rembg` when the input needs foreground extraction. Use `--export_texmap` to export a texture-mapped mesh instead of vertex colors.

### Original InstantMesh baseline

```bash
python run.py configs/instant-mesh-large-lowvram.yaml path/to/main.png --save_video
```

### Separate reranking baseline

The repository also contains a non-training baseline that generates several ordinary Zero123++ sheets and selects one using CLIP similarity to the main image and supplied references. It does **not** inject reference features into Zero123++:

```bash
python run.py configs/instant-mesh-large-lowvram.yaml path/to/main.png \
  --reference_images path/to/references \
  --reference_num_seeds 4 \
  --reference_weight 0.2
```

## Training

### Dataset preparation

The builder can render an object-level dataset from local 3D assets. The command below matches the 500-object configuration used by the recorded run:

```bash
blender --background --python scripts/build_objaverse_reference_dataset.py -- \
  --source_dir data/source_models \
  --output_root data/reference_zero123plus_objaverse_toys_500 \
  --category_keywords plushie toy stuffed_animal doll mascot cartoon_figure animal_toy soft_toy \
  --max_objects 500 \
  --train_ratio 0.9 \
  --val_ratio 0.1 \
  --render_white_background \
  --image_size 320
```

The builder creates object folders plus `train.jsonl`, `val.jsonl`, `split_report.json`, and a preview grid. Each accepted object contains one condition image, six ordered target views, reference views, and pose metadata. It applies occupancy, centering, and image-integrity checks before admitting rendered objects.

Validate and inspect the result with:

```bash
python scripts/validate_reference_adapter_dataset.py \
  --root_dir data/reference_zero123plus_objaverse_toys_500

python scripts/visualize_reference_adapter_sample.py \
  --root_dir data/reference_zero123plus_objaverse_toys_500
```

### Adapter training

The recorded five-epoch configuration uses batch size 1, three references per object, learning rate `1e-5`, up to 450 training objects and 50 validation objects, eight validation batches per epoch, wide pose routing, and adapter-only checkpoints.

```bash
python train.py \
  --base configs/zero123plus-reference-adapter-objaverse-wide-500obj-5epoch-val-loss.yaml \
  --gpus 0 \
  --num_nodes 1
```

Adapter checkpoints are configured at steps 437, 874, 1311, 1748, and 2185, followed by `adapter_last.pt`. Logs and checkpoints are written below `logs/<run-name>/`.

> [!CAUTION]
> A clean checkout is **not currently end-to-end training reproducible**: the configs and validation script import `src.data.objaverse_zero123plus`, but that data module is absent from the tracked repository. Datasets, `logs/`, and checkpoints are also intentionally untracked. Restore the matching data module and provide the rendered dataset before running validation or training. This limitation does not affect inspection of the adapter/routing code, but it blocks a clean-clone training run.

## Evaluation

The ablation script compares ordinary Zero123++, the trained adapter with no references, correct references, an optional earlier checkpoint, and optional incorrect references. Keep the same seed across conditions so changes are attributable to conditioning rather than sampling.

```bash
python scripts/eval_reference_adapter_ablation.py \
  --config configs/instant-mesh-large-lowvram.yaml \
  --input path/to/main.png \
  --reference_images path/to/references \
  --reference_metadata path/to/ref_metadata.json \
  --adapter_last path/to/adapter_last.pt \
  --adapter_step500 path/to/earlier_adapter.pt \
  --output_dir outputs/reference_ablation \
  --seed 42 \
  --zero123plus_pose_version v1.2 \
  --no_rembg
```

Outputs include each condition's 3×2 sheet, a comparison grid, per-slot grids, condition/debug JSON, and CSV/JSON metrics. The implemented metric is mean absolute pixel difference from the baseline, reported globally and for front versus side/back slots. It measures whether and where conditioning changes the output; it is **not** a perceptual-quality or 3D-accuracy score. Despite exposing `--run_reconstruction`, this script deliberately raises `NotImplementedError` for that flag; reconstruct selected sheets separately.

## Results

Only training evidence is committed, so the claims below are deliberately limited to [`training_log.txt`](training_log.txt). No qualitative ablation sheets or mesh-quality benchmark are tracked.

The logged run used 437 training samples and 49 validation samples on one NVIDIA GeForce RTX 3060. It completed five epochs (2,185 optimizer steps; the configured 2,500-step ceiling was not reached because `max_epochs=5`) and saved `adapter_last.pt` in the original training environment.

| Completed epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.150 | 0.159 |
| 2 | 0.129 | 0.122 |
| 3 | 0.105 | **0.0931** |
| 4 | 0.100 | 0.103 |
| 5 | 0.0995 | 0.106 |

The best logged validation loss occurred after epoch 3, while later validation loss increased. Periodic diagnostics reported finite, non-zero reference effects and non-zero gradients for both the projection MLP and view embeddings. These checks show that the adapter was active and trainable; they do not by themselves establish better hidden-view fidelity. The configured post-training smoke test failed because its local input image (`images/nice.jpg`) was absent in that environment.

## Limitations

- Reference discovery/retrieval is out of scope; users must provide references and preferably their camera poses.
- The tracked repository has no trained adapter checkpoint, rendered dataset, evaluation outputs, or example input images.
- The training data module referenced by the adapter configs is missing from version control, so clean-clone training is currently blocked.
- The only committed quantitative evidence is diffusion training/validation loss. There is no tracked baseline-versus-adapter perceptual or geometry result proving an improvement.
- The best validation loss was at epoch 3 rather than the final epoch, suggesting overfitting or normal validation variance; checkpoint selection should be based on held-out evaluation.
- Fixed 3×2 tile masks route influence by output view, not by object part. They cannot isolate a small logo or local feature inside one tile.
- Incorrect references or camera metadata can inject inconsistent appearance into the selected views.
- Objaverse-style held-out objects are useful for internal evaluation but do not establish real-image generalization and may overlap conceptually with data seen by frozen pretrained models.
- Full inference remains GPU- and memory-intensive because Zero123++ and InstantMesh are still large frozen models.

## Future Work

The directions below are planned research, not features of the current repository. They would extend the Zero123++ multi-view generation stage; the original InstantMesh framework would continue to provide the downstream image-to-3D reconstruction pipeline.

### 1. Replace the custom adapter with a true IP-Adapter

> The current reference adapter is inspired by IP-Adapter but does not add fully decoupled image cross-attention throughout the Zero123++ UNet. Future work will replace it with a complete IP-Adapter-style architecture.

The current implementation uses the frozen Zero123++ CLIP image encoder, a two-layer projection MLP, learned view embeddings, and tokens appended to the normal conditioning sequence. Its 2,106,368 trainable parameters (approximately 2.1M) are small relative to the frozen 865M-parameter Zero123++ UNet, but this token-injection design is not a true IP-Adapter.

A future implementation would retain a pretrained image encoder, project its reference features into image-conditioning tokens, and add separate, decoupled image cross-attention layers within the Zero123++ UNet. Original Zero123++ conditioning and reference conditioning would remain separate, with an adjustable reference scale. The base UNet would stay frozen while only the image projection and added image-attention parameters were trained.

### 2. Accept unlabeled references without manual view metadata

The current workflow uses azimuth/elevation metadata, inferred view labels, or filename conventions to build `ref_slot_weights` for the fixed 3×2 sheet. The intended interface is simpler:

```text
Main input image
        +
One or more unlabeled reference images
        ↓
Automatic reference understanding and routing
        ↓
Reference-guided Zero123++ generation
```

Image encoding would still be required. What would disappear from the user-facing workflow is the need to provide azimuth, elevation, view IDs, output-slot assignments, or hand-built slot weights. Candidate approaches include automatic pose or viewpoint estimation, learned view embeddings, feature-based relevance estimation, learned soft routing, and attention-derived spatial masks. The long-term objective is to replace fixed metadata-based routing with a model that estimates whether a reference shows a front, side, back, intermediate view, or useful hidden detail.

### 3. Retrieve useful side and back references automatically

The current project does **not** implement automatic retrieval; it assumes that references have already been found and supplied. A future retrieval stage should find the same object, product, toy, plushie, or character while favoring complementary viewpoints and hidden information—not merely visually similar objects.

```text
Single input image
        ↓
Object or character identification
        ↓
Visual and semantic query generation
        ↓
Search an approved image source or local database
        ↓
Same-object filtering and viewpoint classification
        ↓
Complementary-view usefulness ranking
        ↓
Select side and back references
        ↓
Reference-guided Zero123++ generation
```

Candidate systems could combine CLIP or DINOv2 embeddings, image/text retrieval, local vector search, feature matching, segmentation, duplicate removal, viewpoint classification, and identity-confidence scoring. Ranking should balance two understandable criteria: **identity similarity**, meaning that colors, shape, texture, markings, accessories, and product or character identity agree; and **complementary-view usefulness**, meaning that the candidate exposes a side, back, accessory, pattern, logo, tail, or other detail absent from the input. Different objects, conflicting variants, severe occlusions, unreliable viewpoints, near-duplicates, and candidates with no new information should be rejected.

### Long-term target system

```text
Single input image
        ↓
Identify the object or character
        ↓
Retrieve matching side and back images
        ↓
Filter incorrect and conflicting candidates
        ↓
Estimate reference confidence and viewpoint automatically
        ↓
Encode selected references using a true IP-Adapter
        ↓
Automatically route details to relevant generated views
        ↓
Generate a consistent six-view Zero123++ sheet
        ↓
Reconstruct the final 3D mesh using InstantMesh
```

The long-term goal is to transform the current manually guided prototype into an automatic reference-guided image-to-3D system. A user would provide only one input image, while the system would retrieve useful side and back references, reject incorrect candidates, automatically determine how each reference should influence the generated views, and inject selected information through a true IP-Adapter architecture. The final system would retain the original InstantMesh reconstruction pipeline while improving the Zero123++ hidden-view generation stage.

## Reproduction checklist

1. Install the pinned Python/PyTorch stack and Blender.
2. Obtain the InstantMesh/Zero123++ pretrained weights under their respective terms.
3. Restore `src/data/objaverse_zero123plus.py`, which is referenced but not tracked.
4. Render or otherwise provide the six-view training dataset and validate its manifests.
5. Train the adapter with the five-epoch config and retain all adapter-only checkpoints.
6. Evaluate baseline, no-reference, correct-reference, and optional incorrect-reference conditions with an identical seed.
7. Select a checkpoint using held-out perceptual/geometry results, not training loss alone.
8. Run selected six-view sheets through the frozen InstantMesh reconstruction stage and report both view-level and 3D-level metrics.

## Conclusion

This repository demonstrates a parameter-efficient way to expose Zero123++ to details that a single front image cannot contain. The approach preserves the pretrained Zero123++ and InstantMesh models, trains only a 2.1M-parameter adapter, routes reference information with explicit camera metadata, and spatially limits its effect to relevant views in the fixed output sheet.

The implementation and logged optimization behavior are present, but the repository does not yet contain enough artifacts to claim a measured quality improvement or reproduce training from a clean checkout. The next necessary step is to publish the missing data loader, adapter checkpoint, held-out outputs, and baseline-versus-adapter perceptual and 3D evaluation.

## License and acknowledgements

The repository carries the [Apache License 2.0](LICENSE) inherited from InstantMesh. That file applies to this repository's code distribution; pretrained models and datasets may have separate terms that users must review.

In addition to InstantMesh, this code builds on or acknowledges [Zero123++](https://github.com/SUDO-AI-3D/zero123plus), [OpenLRM](https://github.com/3DTopia/OpenLRM), [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes), and [Instant3D](https://instant-3d.github.io/).
