# Reference-Guided Hidden-View Generation for InstantMesh

This is my PBL5 project. I wanted to see whether extra side and back images could help Zero123++ generate details that are not visible in the main input image.

The project is based on [InstantMesh](https://github.com/TencentARC/InstantMesh). I changed the Zero123++ view-generation stage by adding a small reference adapter, camera-based view routing, and spatial gating. InstantMesh still handles the final 3D reconstruction.

I used Codex for most of the coding. This README explains the approach, the recorded training run, and the parts that still need work.

**Current status:** The adapter has a completed five-epoch training run, but the recorded losses do not show whether it produces better hidden views or meshes. The published repository also has a missing data-loading module, so training cannot be reproduced from a fresh clone yet. See [Running the project](#running-the-project).

## Why I tried this

InstantMesh starts with one image. Zero123++ generates six views, and the reconstruction model uses those views to build a mesh. If the input only shows the front of an object, the model has to guess what the side and back look like.

I wanted to give it more information instead of relying only on that guess. For example, a back reference could show a pattern or detail that is missing from the front image. The question was whether a small adapter could pass that information to the relevant generated views while keeping the main models frozen.

```text
Main input image + side/back references + camera metadata
                            |
          Zero123++ with reference conditioning
                            |
                 Six views in a 3x2 sheet
                            |
             InstantMesh reconstruction model
                            |
                         3D mesh
```

The reference images must be supplied by the user or dataset. Automatic image search and camera-pose estimation are not implemented.

## How it works

1. The frozen CLIP vision encoder in Zero123++ extracts an embedding from each reference image.
2. A two-layer MLP projects the embeddings, and a learned view embedding adds information about the reference view.
3. Camera azimuth and elevation determine how strongly each reference should affect each of the six output views.
4. The adapter produces one weak global token and six view tokens. These are appended to the usual conditioning tokens.
5. With spatial gating enabled, the model computes predictions with and without the reference tokens. A fixed mask controls how much of their difference is applied to each output tile.

The adapter has **2,106,368 trainable parameters**, about 2.1M. In the recorded adapter-only run, the Zero123++ UNet, VAE, vision encoder, and text encoder were frozen. The spatial masks add no trainable parameters. The InstantMesh reconstruction model is not trained in this project.

This is a custom token adapter inspired by image conditioning. It is not a full IP-Adapter with separate image cross-attention layers throughout the UNet.

For Zero123++ v1.2, the output sheet has three rows and two columns:

| Row | Left view | Right view |
|---|---|---|
| 1 | Slot 0: azimuth 30°, elevation 20° | Slot 1: azimuth 90°, elevation -10° |
| 2 | Slot 2: azimuth 150°, elevation 20° | Slot 3: azimuth 210°, elevation -10° |
| 3 | Slot 4: azimuth 270°, elevation 20° | Slot 5: azimuth 330°, elevation -10° |

The masks work on whole view tiles. They cannot isolate a small logo or object part inside a tile, and reference influence can still extend to other views.

## Training and results

The dataset tools render views from 3D assets. The training configuration targets plushie and toy objects in an Objaverse-style dataset; these are rendered images, not a collection of photographs of 500 physical toys.

The recorded run in [`training_log.txt`](training_log.txt) used:

- 437 training samples and 49 validation samples from the dataset prepared for the 500-object experiment.
- Three references per object, batch size 1, and a learning rate of `1e-5`.
- One NVIDIA GeForce RTX 3060.
- Five epochs and 2,185 optimizer steps, with only the adapter trained.
- Wide pose routing, cross-view propagation, and spatial gating.

| Completed epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 0.150 | 0.159 |
| 2 | 0.129 | 0.122 |
| 3 | 0.105 | **0.0931** |
| 4 | 0.100 | 0.103 |
| 5 | 0.0995 | 0.106 |

These are diffusion prediction losses using mean squared error. Validation was limited to **eight batches per epoch**, rather than the full 49-sample validation set. It also samples random noise and timesteps, so these numbers alone are not enough to choose the best model confidently.

Epoch 3 had the lowest recorded validation loss. Training loss continued to decrease after that, while validation loss increased. The run ended at the five-epoch limit and saved `adapter_last.pt`; it did not reach the configured 2,500-step limit.

The log shows non-zero gradients in the adapter and changes in the model predictions. This shows that the adapter was receiving updates, but it does not establish better image quality or more accurate 3D geometry. Generated comparisons and image-quality or geometry results are not included in this repository.

The log uses older `rag_*` names. The current code uses `reference_*` names; the older name does not mean automatic retrieval was implemented.

## Running the project

**The public checkout is incomplete for training.** The config and dataset validator import `src/data/objaverse_zero123plus.py`, but that file is missing from the repository. The training dataset and trained adapter checkpoints are also not included. The commands below document the workflow and require those missing pieces where applicable.

### Environment

The documented setup uses Python 3.10, PyTorch 2.1.0, PyTorch Lightning 2.1.2, and CUDA 12.1. The training and full mesh-generation paths use a CUDA GPU.

Run these commands from the repository root after cloning it:

```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip
conda install ninja
conda install cuda -c nvidia/label/cuda-12.1.0

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7
pip install -r requirements.txt
```

Some dependencies in `requirements.txt` are not pinned, so this is not a fully locked environment. `nvdiffrast` is installed from NVIDIA's Git repository. Blender is needed if you want to render a dataset from 3D assets.

The inference scripts download the Zero123++ and InstantMesh weights if they are not cached. A trained reference adapter checkpoint must be provided separately.

### Reference images

Supply one reference image or a folder of images. Use a JSON file to describe the camera angles, in degrees:

```json
{
  "side.png": {"azimuth": 90, "elevation": -10},
  "back.png": {"azimuth": 180, "elevation": 0}
}
```

The keys must match the reference filenames. The loader can also read poses from filenames such as `ref_180_el0.png`, or use coarse labels such as `front`, `side`, and `back`. Exact camera metadata gives more specific routing. Unknown views receive weak routing.

### Generate a mesh

Replace the example paths with your input, references, metadata, and trained checkpoint:

```bash
python run.py configs/instant-mesh-large-lowvram.yaml path/to/main.png \
  --no_rembg \
  --enable_reference_adapter \
  --reference_images path/to/references \
  --reference_metadata path/to/ref_metadata.json \
  --reference_adapter_ckpt path/to/adapter_last.pt \
  --reference_global_scale 0.1 \
  --reference_spatial_gating \
  --reference_slot_weight_mode wide \
  --reference_cross_view_propagation_enabled \
  --save_video
```

For the baseline without the adapter:

```bash
python run.py configs/instant-mesh-large-lowvram.yaml path/to/main.png \
  --no_rembg --save_video
```

Remove `--no_rembg` if the input needs background removal. Outputs go under `outputs/<config-name>/`, in `images/`, `meshes/`, and `videos/`. Add `--export_texmap` if you want a texture-mapped mesh instead of vertex colors.

### Prepare data and train

The builder accepts local `.glb`, `.gltf`, `.obj`, and `.fbx` assets. This example requests up to 500 objects and a 90/10 train/validation split; the final counts depend on the supplied assets and validation checks.

```bash
blender --background --python scripts/build_objaverse_reference_dataset.py -- \
  --source_dir data/source_models \
  --output_root data/reference_zero123plus_objaverse_toys_500 \
  --max_objects 500 \
  --train_ratio 0.9 \
  --val_ratio 0.1 \
  --render_white_background \
  --image_size 320
```

Use a source folder containing the intended object category. The builder also supports metadata and keyword filtering. It writes input, target, and reference images, pose metadata, split manifests, and a split report.

Once the missing data module has been restored and the dataset is available:

```bash
python scripts/validate_reference_adapter_dataset.py \
  --root_dir data/reference_zero123plus_objaverse_toys_500

python scripts/visualize_reference_adapter_sample.py \
  --root_dir data/reference_zero123plus_objaverse_toys_500

python train.py \
  --base configs/zero123plus-reference-adapter-objaverse-wide-500obj-5epoch-val-loss.yaml \
  --gpus 0 --num_nodes 1
```

The [five-epoch config](configs/zero123plus-reference-adapter-objaverse-wide-500obj-5epoch-val-loss.yaml) saves adapter-only checkpoints at steps 437, 874, 1311, 1748, and 2185, plus `adapter_last.pt`, below `logs/<run-name>/`.

### Compare generated views

The evaluation script compares the baseline, the adapter without references, and the adapter with the supplied references. It resets the seed for each condition.

```bash
python scripts/evaluate_reference_adapter_conditions.py \
  --config configs/instant-mesh-large-lowvram.yaml \
  --input path/to/main.png \
  --reference_images path/to/references \
  --reference_metadata path/to/ref_metadata.json \
  --adapter_last path/to/adapter_last.pt \
  --output_dir outputs/reference_ablation \
  --seed 42 --no_rembg
```

It saves comparison sheets, per-view grids, debug information, and CSV/JSON metrics. The metric is mean absolute pixel difference from the baseline. It measures how much the image changed, not whether it improved. The `--run_reconstruction` option raises `NotImplementedError`.

The comparison script currently uses local routing and a global token scale of `0.05`, while the recorded training run used wide routing, cross-view propagation, and an effective global token scale of `0.1`. Also, inference loads the InstantMesh white-background UNet weights, while the training code loads the base Zero123++ model. These differences need to be accounted for before treating the comparison as a matched evaluation of the training setup.

## Main files

| File | Purpose |
|---|---|
| [`zero123plus/reference_adapter.py`](zero123plus/reference_adapter.py) | Reference projection, view tokens, and tile masks |
| [`zero123plus/reference_utils.py`](zero123plus/reference_utils.py) | Image loading, camera metadata, and view-routing weights |
| [`zero123plus/model.py`](zero123plus/model.py) | Training, frozen modules, and validation loss |
| [`zero123plus/pipeline.py`](zero123plus/pipeline.py) | Reference conditioning during generation |
| [`run.py`](run.py) | View generation followed by InstantMesh reconstruction |
| [`scripts/build_objaverse_reference_dataset.py`](scripts/build_objaverse_reference_dataset.py) | Dataset rendering and manifest creation |
| [`scripts/evaluate_reference_adapter_conditions.py`](scripts/evaluate_reference_adapter_conditions.py) | Comparisons between reference conditions |
| [`training_log.txt`](training_log.txt) | The recorded five-epoch run |

## What I would try next

First, I would restore the missing data module and make the training and evaluation settings consistent. Then I would compare generated views against held-out targets and inspect the reconstructed meshes. Testing with wrong references would also help show how easily the adapter introduces incorrect details.

Beyond that, I am interested in three directions:

- A full IP-Adapter-style approach with separate image cross-attention.
- Estimating the reference viewpoint so the user does not need to provide camera metadata.
- Finding useful side and back images automatically, while checking that they show the same object.

These are possible next steps, not implemented features. The larger idea is still to start with one image, find useful extra views, and use them to reduce missing or incorrect details in the reconstruction.

## Credits and license

The base pipeline, reconstruction model, and much of this repository come from [Tencent ARC's InstantMesh](https://github.com/TencentARC/InstantMesh). The changes here focus on reference conditioning in the Zero123++ stage and the related training, dataset, and evaluation tools.

Please credit the original authors when using their work:

```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

See also [Zero123++](https://github.com/SUDO-AI-3D/zero123plus), [OpenLRM](https://github.com/3DTopia/OpenLRM), [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes), and [Instant3D](https://instant-3d.github.io/).

This repository retains the [Apache 2.0 license](LICENSE) from InstantMesh. Pretrained models and datasets have their own terms.
