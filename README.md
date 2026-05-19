# Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![ArXiv](https://img.shields.io/badge/arXiv-2601.04068-b31b1b.svg)](https://arxiv.org/abs/2601.04068)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://1170300714.github.io/LocalDPO/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/cszthuang/Local_DPO)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


> **Official PyTorch implementation of "Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models" (CVPR 2026).**

## 📢 News
- **[2026.05]** We released the training code, inference code and pretrained lora weights.
- **[2026.02]** Our paper is accepted to **CVPR 2026**! 🎉

## To-Do List

We are actively working on releasing all components of LocalDPO. Stay tuned for updates!

- ✅ **Release Inference Code & test data**: Open-source the inference scripts and test data.
- ✅  **Release Pre-trained Checkpoints**: Full release of LocalDPO fine-tuned weights for CogvideoX-2B, CogvideoX-5B, and Wan2.2-1.3B.
- ✅ **Release Corrupted Video Generation Script**: Code to synthesize locally corrupted videos for constructing preference pairs.
- ✅ **Release Training Code**: Complete training pipeline for LocalDPO.


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
You can follow the following steps to run the inference code:
1. Download the base weights and pretrained checkpoints of CogVideoX-2B, CogVideoX-5B, and Wan2.2-1.3B from [here](https://huggingface.co/cszthuang/Local_DPO)

2. Then, inference on prepared test prompts with pre-trained checkpoints:
   ```bash
   bash local_launch.sh test_base \
      1 \ # number of gpus
      OUTPUT_DIR \
      49 \ # number of frames per video
      720 \ # height
      1280 \ # width    
      1 \ # number of video per prompt
      demo_data/prompt.json \ # prompt file
      BASE_MODEL_PATH \ # the path to the base model weights
      TUNED_MODEL_PATH \ # the path to the tuned model weights (lora)
   ```
3. You can also perform inference on your custom prompts by replacing demo_data/prompt.json with your own. Note that the prompts file itself should be a JSON list, with the specific format as follows: 
   ```json
   [
      {"long": "PROMPT1"},
      {"long": "PROMPT2"},
      ...
   ]
   ```


## 📚 Training Local DPO with Custom Data
You can follow the following steps to generate locally corrupted video and train Local DPO with your own data:
1. Prepare custom real video data and corresponding description, which will be used to generate corrupted data. The meta data of the data should be a JSONL file, with the specific format as follows: 
   ```json
   {"video_path": "PATH_TO_VIDEO1", "description": "DESCRIPTION1 (CAPTION)", "vid": "VIDEO_ID1"},
   {"video_path": "PATH_TO_VIDEO2", "description": "DESCRIPTION2(CAPTION)", "vid": "VIDEO_ID2"},
   ...
   ```
2. Then, generate corrupted video from real video with base model:
   ```bash
   bash local_launch.sh generate_corrupted_video \
      1 \ # number of gpus
      OUTPUT_DIR \
      49 \ # number of frames per video
      720 \ # height
      1280 \ # width    
      REAL_VIDEO_META_DATA \ # your real video meta data
      BASE_MODEL_PATH \ # the path to the base model weights
   ```
   The resized real videos, generated videos and random 3D maskswill be saved in OUTPUT_DIR. The prefixname of each corrupted video and 3D mask are the same as the video's name.
3. Create metadata for Local DPO training data. The metadata should be JSONL file, whose specific format as follows: 
   ```json
   {
      "height_win": "The height of the winner sample (int)", 
      "width_win": "The width of the winner sample (int)", 
      "height_lose": "The height of the loser sample (int)", 
      "width_lose": "The width of the loser sample (int)", 
      "fps_win": "The fps of the winner sample (int)", 
      "fps_lose": "The fps of the loser sample (int)", 
      "duration_win": "The total seconds of the winner sample (float)", 
      "duration_lose": "The total seconds of the loser sample (float)", 
      "pos_num_frames": "The number of frames in the winner sample (int)", 
      "neg_num_frames": "The number of frames in of the loser sample (int)", 
      "pos_video_path": "The path of the winner sample (str)", 
      "neg_video_path": "The path of the loser sample (str)", 
      "mask": "The path of genearted 3D mask of the winner sample (str)", 
      "yita": "The strenth of inversion noise (float)", 
      "gen_caption": "video description (str)"
   },
   {...},
   {...},
   ...
   ```
4. Train model:
   ```bash
   bash local_launch.sh train_base \
      1 \ # number of gpus
      OUTPUT_DIR \
      META_DATA_PATH \ # your metadata
      BASE_MODEL_PATH \ # the path to the base model weights
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


## Contact
If you have any questions, please contact us via
- 1017141005@qq.com
- richu@mail.ustc.edu.cn

## 🙏  Acknowledgements
This codebase builds upon several excellent open-source projects:
 - [diffusers](https://github.com/huggingface/diffusers)
 - [cogvideoX](https://github.com/zai-org/CogVideo)
 - [Wan2.1](https://github.com/Wan-Video/Wan2.1)