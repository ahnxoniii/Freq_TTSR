# TTSR++  

This repository is the official PyTorch implementation of **"Improved Reference-Based Super-Resolution with Frequency Information Fusion"**  
This paper published 한국멀티미디어학회논문지 (ISSN: 1229-7771), 제29권, 2호, 2026.02
https://doi.org/10.9717/kmms.2026.29.2.183


----
Soeun An, Hanhoon Park(Major Professor)  
**IVC(Image & Vision Computing) Lab / Pukyong Nat'l Univ Electronic Engineering / Busan, Republic of Korea**  

## Abstract
As one of the super-resolution (SR) technologies, reference-based super-resolution (RefSR) has
greatly improved SR performance by using high-quality reference images as inputs. However, existing
RefSR methods have limitations in that global texture and structure information is not sufficiently
transferred because they transfer and fuse texture information of reference images based on the similarity
between LR-reference images in the spatial domain. To address this problem, this paper proposes
methods for transferring and fusing texture information of reference images in the frequency domain.
By applying the proposed methods to the existing RefSR method, TTSR, we show that complementary
texture information in the spatial and frequency domains can be transferred and utilized together, which
can significantly improve SR performance. In particular, the PieAPP value, a perceptual quality indicator
of SR images, has improved by more than 10%.

## Results
<img width="631" height="616" alt="image" src="https://github.com/user-attachments/assets/242f8558-beab-48b1-9e66-cfb9d953d362" />
<img width="408" height="118" alt="image" src="https://github.com/user-attachments/assets/19afaea4-70b0-4a0d-8928-b2567583c87c" />

##  Prerequisites 🛠️
Follow the steps below to set up the environment and install dependencies.  
Ubuntu 20.04, CUDA 
1. Clone this github repo
```bash
git clone https://github.com/ahnxoniii/TTSR_plus.git
cd TTSR_plus
```
2. Create environment  
We recommend using an **Anaconda environment**.
```bash
conda create -n TTSRplus_env python=3.7 -y
conda activate TTSRplus_env
```
3. Install python dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation
You can download **CUFED5** here : https://github.com/ZZUTK/SRNTT
```
├── CUFED
    ├── train
        ├── input
        ├── ref
    ├── test
        ├── CUFED5
```

## Train
```bash
sh train.sh
```

## Eval
```bash
sh eval.sh
```



