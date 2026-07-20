<div align="center">
  
# InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models

<a href="https://arxiv.org/abs/2404.07191"><img src="https://img.shields.io/badge/ArXiv-2404.07191-brightgreen"></a> 
<a href="https://huggingface.co/TencentARC/InstantMesh"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> 
<a href="https://huggingface.co/spaces/TencentARC/InstantMesh"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a> <br>
<a href="https://replicate.com/camenduru/instantmesh"><img src="https://img.shields.io/badge/Demo-Replicate-blue"></a>
<a href="https://colab.research.google.com/github/camenduru/InstantMesh-jupyter/blob/main/InstantMesh_jupyter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<a href="https://github.com/jtydhr88/ComfyUI-InstantMesh"><img src="https://img.shields.io/badge/Demo-ComfyUI-8A2BE2"></a>

</div>

---

This repo is the official implementation of InstantMesh, a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

https://github.com/TencentARC/InstantMesh/assets/20635237/dab3511e-e7c6-4c0b-bab7-15772045c47d

# 🚩 Features and Todo List
- [x] 🔥🔥 Release Zero123++ fine-tuning code. 
- [x] 🔥🔥 Support for running gradio demo on two GPUs to save memory.
- [x] 🔥🔥 Support for running demo with docker. Please refer to the [docker](docker/) directory.
- [x] Release inference and training code.
- [x] Release model weights.
- [x] Release huggingface gradio demo. Please try it at [demo](https://huggingface.co/spaces/TencentARC/InstantMesh) link.
- [ ] Add support for more multi-view diffusion models.

# ⚙️ Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# Install other requirements
pip install -r requirements.txt
```

# 💫 How to Use

## Download the models

We provide 4 sparse-view reconstruction model variants and a customized Zero123++ UNet for white-background image generation in the [model card](https://huggingface.co/TencentARC/InstantMesh).

Our inference script will download the models automatically. Alternatively, you can manually download the models and put them under the `ckpts/` directory.

By default, we use the `instant-mesh-large` reconstruction model variant.

## Start a local gradio demo

To start a gradio demo in your local machine, simply run:
```bash
python app.py
```

If you have multiple GPUs in your machine, the demo app will run on two GPUs automatically to save memory. You can also force it to run on a single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```

Alternatively, you can run the demo with docker. Please follow the instructions in the [docker](docker/) directory.

## Running with command line

To generate 3D meshes from images via command line, simply run:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video
```

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object. If the input image already has an alpha mask, please specify the `no_rembg` flag:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video --no_rembg
```

By default, our script exports a `.obj` mesh with vertex colors, please specify the `--export_texmap` flag if you hope to export a mesh with a texture map instead (this will cost longer time):
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video --export_texmap
```

To use retrieved references safely, use RAG reranking. This generates several normal Zero123++ candidate sheets with different seeds, scores them with CLIP against the input and references, then sends only the selected sheet to InstantMesh:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --rag_refs path/to/reference_images --rag_num_seeds 4 --rag_weight 0.2
```
Reference images are never injected into Zero123++ conditioning. The fixed 6-view Zero123++ layout and InstantMesh reconstruction input format are preserved. Include `front`, `side`, or `back` in reference filenames, or provide `--rag_view_labels labels.json`, to guide view-aware scoring.

To create a tiny RAG adapter training set from local 3D assets, place `.glb`, `.obj`, or `.fbx` files in `data/source_models/`, then run Blender:
```bash
blender --background --python scripts/render_rag_zero123plus_tiny.py -- --source_dir data/source_models --output_dir data/rag_zero123plus_tiny --max_objects 10
python scripts/validate_rag_adapter_dataset.py --root_dir data/rag_zero123plus_tiny
python scripts/visualize_rag_adapter_sample.py --root_dir data/rag_zero123plus_tiny
```

To scale the RAG adapter dataset with local Objaverse or Objaverse-style toy/plushie assets, use the object-level dataset builder. It does not download large assets automatically; provide a local model folder or metadata file with local paths:
```bash
python scripts/build_objaverse_rag_dataset.py \
  --output_root data/rag_zero123plus_objaverse_toys \
  --source_dir data/source_models \
  --category_keywords plushie toy stuffed_animal doll mascot cartoon_figure animal_toy soft_toy \
  --max_objects 1000 \
  --train_ratio 0.9 \
  --val_ratio 0.1 \
  --render_white_background \
  --image_size 320
```
When rendering is needed, run the same script through Blender:
```bash
blender --background --python scripts/build_objaverse_rag_dataset.py -- --source_dir data/source_models --output_root data/rag_zero123plus_objaverse_toys --max_objects 1000 --render_white_background
```
The builder writes object-level `train.jsonl`, `val.jsonl`, optional `test_objaverse_heldout.jsonl`, `split_report.json`, `dataset_preview_grid.png`, accepted objects under `objects/`, rejected examples under `rejected/`, and an `external_test/` folder for later manually collected real plushie/toy examples. Train the larger adapter variant with:
```bash
python train.py --base configs/zero123plus-rag-adapter-objaverse-wide-1000.yaml --gpus 0 --num_nodes 1
```

For a medium 500-object / 500-step validation-enabled experiment, build a separate dataset root and train with the dedicated config:
```bash
python scripts/build_objaverse_rag_dataset.py \
  --output_root data/rag_zero123plus_objaverse_toys_500 \
  --category_keywords plushie toy stuffed_animal doll mascot cartoon_figure animal_toy soft_toy \
  --max_objects 500 \
  --train_ratio 0.9 \
  --val_ratio 0.1 \
  --render_white_background \
  --image_size 320

python train.py --base configs/zero123plus-rag-adapter-objaverse-wide-500obj-500steps-val.yaml --gpus 0 --num_nodes 1
```

Objaverse is useful for scalable adapter training, but evaluation only on Objaverse-style objects may be biased because the frozen Zero123++ base model may have seen similar data during pretraining. Therefore, Objaverse held-out evaluation should be treated as internal evaluation, not final generalization proof. Keep final generalization checks separate by evaluating manually collected real plushie/toy examples in `data/rag_zero123plus_objaverse_toys/external_test/`.

To evaluate a trained view-aware reference-token adapter with spatial gating, run the Zero123++ ablation script. It saves each 3x2 sheet, a side-by-side grid, per-slot comparisons, debug logs, reference slot weights, and simple difference metrics without running InstantMesh reconstruction:
```bash
python scripts/eval_rag_adapter_ablation.py --config configs/instant-mesh-large-lowvram.yaml --input images/nice.jpg --rag_refs folder --rag_ref_metadata folder/ref_metadata.json --adapter_last logs/zero123plus-rag-adapter-100/adapter_checkpoints/adapter_last.pt --adapter_step500 logs/zero123plus-rag-adapter-100/adapter_checkpoints/adapter_step_000500.pt --output_dir outputs/rag_ablation --seed 42 --zero123plus_pose_version v1.2 --no_rembg
```

Please use a different `.yaml` config file in the [configs](./configs) directory if you hope to use other reconstruction model variants. For example, using the `instant-nerf-large` model for generation:
```bash
python run.py configs/instant-nerf-large.yaml examples/hatsune_miku.png --save_video
```
**Note:** When using the `NeRF` model variants for image-to-3D generation, exporting a mesh with texture map by specifying `--export_texmap` may cost long time in the UV unwarping step since the default iso-surface extraction resolution is `256`. You can set a lower iso-surface extraction resolution in the config file.

# 💻 Training

We provide our training code to facilitate future research. But we cannot provide the training dataset due to its size. Please refer to our [dataloader](src/data/objaverse.py) for more details.

To train the sparse-view reconstruction models, please run:
```bash
# Training on NeRF representation
python train.py --base configs/instant-nerf-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1

# Training on Mesh representation
python train.py --base configs/instant-mesh-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```

We also provide our Zero123++ fine-tuning code since it is frequently requested. The running command is:
```bash
python train.py --base configs/zero123plus-finetune.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```

# :books: Citation

If you find our work useful for your research or applications, please cite using this BibTeX:

```BibTeX
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

# 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
- [Instant3D](https://instant-3d.github.io/)

Thank [@camenduru](https://github.com/camenduru) for implementing [Replicate Demo](https://replicate.com/camenduru/instantmesh) and [Colab Demo](https://colab.research.google.com/github/camenduru/InstantMesh-jupyter/blob/main/InstantMesh_jupyter.ipynb)!  
Thank [@jtydhr88](https://github.com/jtydhr88) for implementing [ComfyUI support](https://github.com/jtydhr88/ComfyUI-InstantMesh)!
