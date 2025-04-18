# Synthetic Diffusion Tensor Imaging (DTI) Maps

This project focuses on generating synthetic Diffusion Tensor Imaging (DTI) maps using 2D and 3D probabilistic diffusion models. It leverages Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA and SV-DIFF to enable memory-optimized training and high-fidelity image synthesis on the CamCAN dataset.

## 🧠 Overview

- Trained 2D and 3D diffusion models to synthesize realistic DTI brain maps.
- Evaluated synthetic outputs for structural and anatomical accuracy.
- Integrated PEFT methods (LoRA, SV-DIFF, DiffFit) for efficient fine-tuning.
- Optimized model parameters using bi-level optimization with Optuna.

## 🛠️ Tools & Technologies

- **CamCAN Dataset** (DTI brain scans)
- **Diffusion Models**: 2D & 3D probabilistic architectures
- **PEFT**: LoRA, SV-DIFF, DiffFit
- **Optimization**: Optuna (bi-level search)
- **Frameworks**: PyTorch, MONAI, SimpleITK, nibabel

## 📁 Project Structure
├── data/ # DICOM inputs and preprocessed NIfTI masks ├── models/ # 2D and 3D diffusion model definitions ├── training/ # Training scripts with PEFT integration ├── evaluation/ # Evaluation metrics & anatomical accuracy checks ├── utils/ # Data loading, visualization, and helper functions └── README.md # You're here!


## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/synthetic-dti-maps.git
   cd synthetic-dti-maps
   
2. **Install dependencies:**
   pip install -r requirements.txt

3. **Prepare dataset:**
   Place DICOM files in data/DICOM_input/
   Place corresponding ground-truth masks in data/Masks/

4. **Run training:**
   python training/train_unet.py --mode 3D --peft lora

5.**Evaluate results:**
  python evaluation/eval_metrics.py

## 📊 Results
- Generated DTI maps closely match ground-truth in FA/MD values.
- PEFT models achieved high anatomical fidelity with minimal memory overhead.

## 📄 Publication
This work is part of the publication:
"Synthetic Diffusion Tensor Imaging Maps Generated by 2D and 3D Probabilistic Diffusion Models: Evaluation and Applications"
https://pmc.ncbi.nlm.nih.gov/articles/PMC11888198/



