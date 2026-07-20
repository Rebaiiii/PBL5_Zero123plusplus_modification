# Reference-Guided Hidden-View Generation for InstantMesh

_A small adapter that adds side and back reference details to Zero123++ multi-view generation._


> This project is based on the original [InstantMesh](https://github.com/TencentARC/InstantMesh) project. InstantMesh and the original code were made by the InstantMesh authors. My changes are in the Zero123++ multi-view stage, where I added reference conditioning and spatial gating.

## Overview

[InstantMesh](https://github.com/TencentARC/InstantMesh) creates a 3D object from one image. First, Zero123++ generates six views in a fixed 3×2 sheet. Then the frozen InstantMesh reconstruction model uses these views to create a mesh. When only a front image is given, the model has to guess the side and back details. Because of this, some details can be missing, wrong, or inconsistent.

This project adds side and back reference images to this stage. A small trainable adapter changes the frozen CLIP image features into conditioning tokens. Camera azimuth and elevation are used to decide which generated views need each reference. A spatial mask keeps the reference effect mainly inside the matching tiles.

```text
Main input image
        ↓
Zero123++ with spatially gated reference conditioning <- Reference Adapter <- Additional Reference Image + Camera view metadata
        ↓
Six-view 3×2 sheet
        ↓
InstantMesh reconstruction model
        ↓
3D mesh
```

The main work is **reference-guided hidden-view generation for Zero123++ using a small adapter, camera metadata, and spatial gating**. This is not an automatic retrieval system. The user or dataset must provide the reference images and camera metadata.

## Based on InstantMesh

The image-to-3D pipeline, reconstruction model, project structure, and most supporting parts come from Tencent ARC's InstantMesh. I do not claim ownership of the original InstantMesh architecture or code.

My changes include the reference adapter, view-aware conditioning, spatial gating, adapter-only training and checkpoints, dataset tools, and evaluation tools. Please cite the original InstantMesh paper and repository when using this work. Also follow the licenses for InstantMesh, Zero123++, pretrained models, and datasets.

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

The adapter is small and works as follows:

1. The frozen Zero123++ CLIP vision encoder extracts one embedding per reference image.
2. `ReferenceAdapter` applies a two-layer MLP and adds a learned embedding for the reference view ID.
3. Pose metadata produces six routing weights, one for each fixed Zero123++ target pose.
4. The adapter creates one weak global token and six routed view tokens. These tokens are added to the normal prompt embeddings.
5. Zero123++ runs one normal branch and one reference branch. Their difference is mixed using a fixed tile mask, so the reference mainly changes the related output views.

For Zero123++ v1.2, the row-major sheet layout is:

| Sheet position | Slot | Azimuth | Elevation |
|---|---:|---:|---:|
| Row 1, left | 0 | 30° | 20° |
| Row 1, right | 1 | 90° | −10° |
| Row 2, left | 2 | 150° | 20° |
| Row 2, right | 3 | 210° | −10° |
| Row 3, left | 4 | 270° | 20° |
| Row 3, right | 5 | 330° | −10° |

Spatial gating does not add trainable parameters. During adapter-only training, the Zero123++ UNet, VAE, vision encoder, and text encoder stay frozen. `reference_unfreeze_crossattn` is disabled. The InstantMesh reconstruction model is used after view generation and is not trained here.

## Model size

The model summary in [`training_log.txt`](training_log.txt) shows these parameter counts:

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
| [`run.py`](run.py) | Full Zero123++ → InstantMesh inference with optional reference-adapter conditioning |
| [`scripts/build_objaverse_reference_dataset.py`](scripts/build_objaverse_reference_dataset.py) | Rendering and manifest construction from local 3D assets or Objaverse-style metadata |
| [`scripts/render_single_reference_example.py`](scripts/render_single_reference_example.py) | Render one model into an input image and three reference views for a quick inference example |
| [`scripts/render_reference_training_dataset.py`](scripts/render_reference_training_dataset.py) | Batch-render models into condition, six-target, reference, metadata, and quality files for training |
| [`scripts/evaluate_reference_adapter_conditions.py`](scripts/evaluate_reference_adapter_conditions.py) | Compare reference-adapter conditions and report per-slot pixel differences |

## Installation

The InstantMesh setup uses Python 3.10, PyTorch 2.1.0, and CUDA 12.1. Training and full inference need a CUDA GPU.

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

The requirements install `nvdiffrast` from NVIDIA's Git repository. Blender is also needed to render training images from `.glb`, `.gltf`, `.obj`, or `.fbx` files.

The scripts download `sudo-ai/zero123plus-v1.2` and the InstantMesh weights if they are not cached. You can also put the required InstantMesh files in `ckpts/`. For example, the low-VRAM config uses `diffusion_pytorch_model.bin` and `instant_mesh_large.ckpt`.

## Reference input and metadata

You can give one reference image or a folder of images. Camera metadata is recommended because automatic view assignment is not implemented. The metadata file maps each image name to azimuth and elevation:

```json
{
  "side.png": {"azimuth": 90, "elevation": -10},
  "back.png": {"azimuth": 180, "elevation": 0}
}
```

If there is no metadata file, the pose can come from a filename such as `ref_180_el0.png`. The code can also use simple labels such as `front`, `side`, or `back` from filenames or `--reference_view_labels`. These labels are less accurate than exact camera angles. Unknown views receive only weak routing.

## Inference

### Full reference-guided mesh generation

For reference-guided inference, provide a trained adapter checkpoint with `--reference_adapter_ckpt`.

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

## Training

### Dataset preparation

The dataset builder can render images from local 3D models. This command matches the 500-object training run:

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

The builder creates object folders, `train.jsonl`, `val.jsonl`, `split_report.json`, and a preview grid. Each object has one input image, six target views, reference views, and pose metadata. The builder also checks image size, object position, and whether the files are valid.

Validate and inspect the result with:

```bash
python scripts/validate_reference_adapter_dataset.py \
  --root_dir data/reference_zero123plus_objaverse_toys_500

python scripts/visualize_reference_adapter_sample.py \
  --root_dir data/reference_zero123plus_objaverse_toys_500
```

### Adapter training

The five-epoch run used batch size 1, three references per object, learning rate `1e-5`, up to 450 training objects, and 50 validation objects. It used eight validation batches per epoch, wide pose routing, and adapter-only checkpoints.

```bash
python train.py \
  --base configs/zero123plus-reference-adapter-objaverse-wide-500obj-5epoch-val-loss.yaml \
  --gpus 0 \
  --num_nodes 1
```

Adapter checkpoints are configured at steps 437, 874, 1311, 1748, and 2185, followed by `adapter_last.pt`. Logs and checkpoints are written below `logs/<run-name>/`.


## Evaluation

The evaluation script compares normal Zero123++, the adapter without references, and the adapter with correct references. Use the same seed for all conditions so the comparison is fair.

```bash
python scripts/evaluate_reference_adapter_conditions.py \
  --config configs/instant-mesh-large-lowvram.yaml \
  --input path/to/main.png \
  --reference_images path/to/references \
  --reference_metadata path/to/ref_metadata.json \
  --adapter_last path/to/adapter_last.pt \
  --output_dir outputs/reference_ablation \
  --seed 42 \
  --zero123plus_pose_version v1.2 \
  --no_rembg
```

The script saves each 3×2 sheet, a comparison grid, per-view grids, debug JSON, and CSV/JSON metrics. The current metric is mean absolute pixel difference from the baseline. It shows whether the adapter changed the image and where the change happened. It does not measure image quality or 3D accuracy. The `--run_reconstruction` option is not implemented, so reconstruct selected sheets separately.

## Results

The results below come from [`training_log.txt`](training_log.txt). Image comparisons, reconstructed meshes, and perceptual or geometry metrics are still needed for a full evaluation.

The run used 437 training samples and 49 validation samples on one NVIDIA GeForce RTX 3060. It finished five epochs and 2,185 optimizer steps. It stopped before 2,500 steps because `max_epochs=5`. The final checkpoint was saved as `adapter_last.pt`.

| Completed epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.150 | 0.159 |
| 2 | 0.129 | 0.122 |
| 3 | 0.105 | **0.0931** |
| 4 | 0.100 | 0.103 |
| 5 | 0.0995 | 0.106 |

Epoch 3 had the lowest validation loss. The validation loss increased in epochs 4 and 5. The logs also show non-zero reference changes and gradients for the projection MLP and view embeddings. This confirms that the adapter was training, but it does not prove that the generated hidden views are better.

## Limitations

- The user must find the reference images and provide their camera poses.
- A trained adapter checkpoint, input images, and the pretrained Zero123++ and InstantMesh weights are needed for inference.
- Training needs `src/data/objaverse_zero123plus.py` and dataset manifests that match the config.
- The only numerical results are training and validation loss. Image-quality and 3D metrics are still needed.
- Epoch 3 had a lower validation loss than the final epoch. A validation set should be used to choose the checkpoint.
- The fixed 3×2 masks work by view tile, not by object part. They cannot target only a small logo or detail inside one tile.
- A wrong reference or wrong camera angle can add incorrect details to the output.
- Objaverse-style validation is useful for this project, but it does not prove that the model works well on new real images. It may also be similar to data used by Zero123++.
- Full inference needs a lot of GPU memory because Zero123++ and InstantMesh are large models.

## Future Work

The items below are future plans. They are not implemented now. These changes will be made in the Zero123++ view-generation stage. InstantMesh will still be used for 3D reconstruction.

### 1. Replace the custom adapter with a true IP-Adapter

The current adapter uses the frozen Zero123++ CLIP image encoder, a two-layer MLP, learned view embeddings, and extra conditioning tokens. It has 2,106,368 trainable parameters, or about 2.1M. The frozen Zero123++ UNet has about 865M parameters. This token method is not a true IP-Adapter.

In the future, I want to use a pretrained image encoder and project its features into image tokens. I also want to add separate image cross-attention layers inside the Zero123++ UNet. The normal Zero123++ conditioning and reference conditioning will stay separate. The reference strength will be adjustable. The main UNet will stay frozen, and only the new projection and attention parts will be trained.

### 2. Accept unlabeled references without manual view metadata

The current method uses azimuth, elevation, view labels, or filenames to create `ref_slot_weights`. In the future, the input should be simpler:

```text
Main input image
        +
One or more unlabeled reference images
        ↓
Automatic reference understanding and routing
        ↓
Reference-guided Zero123++ generation
```

The system will still need an image encoder. However, the user should not need to enter azimuth, elevation, view IDs, slot IDs, or slot weights. Possible methods include camera-pose estimation, viewpoint classification, learned view embeddings, soft routing, or learned masks. The model should understand if a reference shows the front, side, back, an angle between them, or a useful hidden detail.

### 3. Retrieve useful side and back references automatically

Automatic reference search is not implemented. At the moment, the user must find the reference images. In the future, the system should search for the same object, product, toy, plushie, or character. It should prefer side or back images that show new details, not only similar-looking images.

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
Rank views by how much new information they show
        ↓
Select side and back references
        ↓
Reference-guided Zero123++ generation
```

This may use CLIP or DINOv2 features, image or text search, feature matching, segmentation, duplicate removal, and viewpoint classification. The search should check two things. First, the image should show the same object. Color, shape, texture, pattern, logo, and accessories can help with this. Second, the image should add useful information, such as a side, back, tail, logo, or hidden pattern. Images with a different object, different version, heavy blocking, unclear viewpoint, or no new details should be rejected.

### Better System

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

The final goal is an automatic reference-guided image-to-3D system. The user will give only one image. The system will find useful side and back references, remove wrong results, decide where each reference is useful, and add the information through a true IP-Adapter. InstantMesh will still do the final 3D reconstruction.

## Reproduction checklist

1. Install the pinned Python/PyTorch stack and Blender.
2. Obtain the InstantMesh/Zero123++ pretrained weights under their respective terms.
3. Provide a compatible `src/data/objaverse_zero123plus.py` data module.
4. Render or otherwise provide the six-view training dataset and validate its manifests.
5. Train the adapter with the five-epoch config and keep all adapter-only checkpoints.
6. Evaluate baseline, no-reference, and correct-reference conditions with an identical seed.
7. Select a checkpoint using held-out perceptual/geometry results, not training loss alone.
8. Run selected six-view sheets through the frozen InstantMesh reconstruction stage and report both view-level and 3D-level metrics.

## Conclusion

This project gives Zero123++ extra side and back information that is not visible in one front image. Zero123++ and InstantMesh stay frozen. Only the 2.1M-parameter adapter is trained. Camera metadata sends the reference information to the related views, and spatial gating limits the change to those view tiles.

## License and acknowledgements

This repository uses the [Apache License 2.0](LICENSE) from InstantMesh. Pretrained models and datasets may have different licenses, so check their terms before use.

This project also uses or refers to [Zero123++](https://github.com/SUDO-AI-3D/zero123plus), [OpenLRM](https://github.com/3DTopia/OpenLRM), [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes), and [Instant3D](https://instant-3d.github.io/).
