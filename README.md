# Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![ArXiv](https://img.shields.io/badge/arXiv-2601.04068-b31b1b.svg)](https://arxiv.org/abs/2601.04068)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://1170300714.github.io/LocalDPO/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


> **Official PyTorch implementation of "Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models" (CVPR 2026).**

## 📢 News
- **[2026.05]** We released the inference code.
- **[2026.02]** Our paper is accepted to **CVPR 2026**! 🎉

## To-Do List

We are actively working on releasing all components of DLDPO. Stay tuned for updates!

- ✅ **Release Inference Code & test data**: Open-source the inference scripts and test data.
- [ ] **Release Pre-trained Checkpoints**: Full release of LocalDPO fine-tuned weights for CogvideoX-2B, CogvideoX-5B, and Wan2.2-1.3B.
- [ ] **Release Corrupted Video Generation Script**: Code to synthesize locally corrupted videos for constructing preference pairs.
- [ ] **Release Training Code**: Complete training pipeline for LocalDPO.
- [ ] **Release Curated Pexels Dataset**: The subset of Pexels videos collected and annotated for local detail preference learning.


## 📖 Abstract

Aligning text-to-video diffusion models with human preferences is crucial for generating high-quality videos. Existing Direct Preference Otimization (DPO) methods rely on multi-sample ranking and task-specific critic models, which is inefficient and often yields ambiguous global supervision. To address these limitations, we propose **LocalDPO**, a novel post-training framework that constructs localized preference pairs from real videos and optimizes alignment at the spatio-temporal region level. We design an automated pipeline to efficiently collect preference pair data that generates preference pairs with a single inference per prompt, eliminating the need for external critic models or manual annotation. Specifically, we treat high-quality real videos as positive samples and generate corresponding negatives by locally corrupting them with random spatio-temporal masks and restoring only the masked regions using the frozen base model. During training, we introduce a region-aware DPO loss that restricts preference learning to corrupted areas for rapid convergence. Experiments on Wan2.1 and CogVideoX demonstrate that LocalDPO consistently improves video fidelity, temporal coherence and human preference scores over other post-training approaches, establishing a more efficient and fine-grained paradigm for video generator alignment.


## 🛠️ Installation

### Prerequisites
- Python >= 3.10
- PyTorch == 2.2.0 

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/1170300714/Local-DPO.git
   cd Local-DPO
   ```
2. **Create a conda environment:**
   ```bash
    conda create -n localdpo python=3.10
    conda activate localdpo
   ```
3. **Install dependencies::**
   ```bash
    pip install -r requirements.txt
   ```

## 🚀 Quick Start
1. **Inference with pre-trained checkpoints:**
   ```bash
    bash local_launch.sh test \
    1 \ # number of gpus
    OUTPUT_DIR \
    49 \ # number of frames per video
    720 \ # height
    1280 \ # width    
    1 \ # number of video per prompt
    demo_data/prompt.json \ # prompt file
    BASE_MODEL_PATH \
    TUNED_MODEL_PATH \ # the path to the tuned model (lora)

   ```

## 📝  Citation
```bibtex
@article{huang2026mind,
  title={Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models},
  author={Huang, Zitong and Zhang, Kaidong and Ding, Yukang and Gao, Chao and Ding, Rui and Chen, Ying and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2601.04068},
  year={2026}
}
```

## 🙏  Acknowledgements
This codebase builds upon several excellent open-source projects:
 - [diffusers](https://github.com/huggingface/diffusers)
 - [cogvideoX](https://github.com/zai-org/CogVideo)
 - [Wan2.1](https://github.com/Wan-Video/Wan2.1)