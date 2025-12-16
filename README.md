# **OODFace**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv:2412.02479-b31b1b.svg)](https://arxiv.org/pdf/2412.02479)

This repository contains the implementation of the paper "[OODFace: Benchmarking Robustness of Face Recognition under Open-World Corruptions and Variations](https://arxiv.org/pdf/2412.02479)" by Caixin Kang, Yubo Chen, Shouwei Ruan, Shiji Zhao, Ruochen Zhang, Jiayi Wang, Shan Fu, Xingxing Wei.

![OODFace Categories](./category.png)

## 📖 Introduction

With the rise of deep learning, facial recognition technology has seen extensive research and rapid development. Although facial recognition is considered a mature technology, we find that existing open-source models and commercial algorithms lack robustness in certain open-world Out-of-Distribution (OOD) scenarios, raising concerns about the reliability of these systems.

In this paper, we introduce **OODFace**, which explores the OOD challenges faced by facial recognition models from two perspectives: **Common Corruptions** and **Appearance Variations**. We systematically design 30 OOD dynamic scenarios across 9 major categories tailored for facial recognition. By simulating these challenges on public datasets, we establish three robustness benchmarks: **LFW-C/V**, **CFP-FP-C/V**, and **YTF-C/V**.


## ✨ Key Features

- **Comprehensive OOD Scenario Coverage**: Systematically designed 30 OOD dynamic scenarios across 9 major categories
- **Dual Challenge Perspectives**:
  - **Common Corruptions**: Including noise, blur, occlusion, compression, and other image quality degradations
  - **Appearance Variations**: Including makeup, hairstyle, glasses, age, and other appearance attribute changes


## 📁 Project Structure

```
OODFace/
├── common_corruptions/          # Common corruptions module
│   └── test_corruptions_demo.py  # Corruption testing demo script
├── appearance_variations/       # Appearance variations module
│   ├── BeautyGAN/               # Makeup transfer tool
│   ├── HiSD/                    # Hierarchical Style Disentanglement for image translation
│   └── PTI/                     # Pivotal Tuning Inversion
├── utils/                       # Utility functions
│   ├── align.py                 # Face alignment
│   ├── face_preprocess.py       # Face preprocessing
│   └── test_demo.py             # Testing demo
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### Requirements

- Python 3.6+
- PyTorch 1.0+ / TensorFlow 1.9+ (depending on the module used)
- OpenCV
- NumPy
- PIL/Pillow
- imagecorruptions (for common corruptions)

### Installation

```bash
# Basic dependencies
pip install opencv-python numpy pillow scikit-image imagecorruptions

# Install dependencies for appearance variation tools as needed
# BeautyGAN requires TensorFlow 1.9
# HiSD requires PyTorch 1.0+
# PTI requires PyTorch and related dependencies
```

### Usage Examples

#### 1. Common Corruptions

```python
from common_corruptions.test_corruptions_demo import pix_process
import cv2

# Read image
image = cv2.imread('input.jpg')

# Apply corruption (supports multiple types)
corrupted_image = pix_process(
    image, 
    distortion_type='gaussian_noise',  # Corruption type
    severity=3  # Severity level (1-5)
)

# Save result
cv2.imwrite('output.jpg', corrupted_image)
```

**Supported Corruption Types**:
- Noise: `gaussian_noise`, `impulse_noise`, `shot_noise`, `speckle_noise`, `salt_and_pepper_noise`
- Blur: `defocus_blur`, `motion_blur`, `zoom_blur`, `frost_blur`, `spatter_blur`, `glass_blur`
- Weather: `fog`, `snow`
- Others: `brightness`, `contrast`, `jpeg_compression`, `random_occlusion`, `color_shift`, `saturate`, `pixelate`, `random_hue_shift`

#### 2. Appearance Variations

##### BeautyGAN - Makeup Transfer

```bash
cd appearance_variations/BeautyGAN
python main.py --no_makeup path/to/image.jpg
```

##### HiSD - Hairstyle/Glasses Attribute Editing

```bash
cd appearance_variations/HiSD
python easy_use.py
# Or use Jupyter Notebook
jupyter notebook easy_use.ipynb
```

##### PTI - Advanced Face Editing

Please refer to `appearance_variations/PTI/README.md` for detailed usage instructions.

#### 3. Face Recognition Model Testing

```bash
# Evaluate models using the test script
python utils/test_demo.py --model MobileFace --log log/log.txt
```

## 📊 Benchmarks

OODFace establishes robustness benchmarks on the following datasets:

- **LFW-C/V**: Corrupted/Variation versions of Labeled Faces in the Wild
- **CFP-FP-C/V**: Corrupted/Variation versions of Celebrities in Frontal-Profile
- **YTF-C/V**: Corrupted/Variation versions of YouTube Faces

Each benchmark includes:
- **C (Corruptions)**: Common corruption versions
- **V (Variations)**: Appearance variation versions


## 📚 Dataset Preparation

### LFW Dataset

1. Download the LFW dataset
2. Use `utils/align.py` for face alignment
3. Apply corruptions or variations to generate OOD versions

### CFP-FP and YTF Datasets

Similarly, other datasets can be processed following the same workflow.

## 🛠️ Extended Usage

### Adding Custom Corruption Types

Add new corruption functions in `common_corruptions/test_corruptions_demo.py` and register them in the `pix_process` function.

### Adding Custom Appearance Variations

Other image editing tools can be integrated into the `appearance_variations/` directory.

### Extending to Other Datasets

The toolkit is designed to be easily extensible to other datasets. Main steps:
1. Prepare the dataset
2. Use `utils/align.py` for face alignment
3. Apply corruptions or variations
4. Use `utils/test_demo.py` for evaluation

## 📝 Citation

If you use OODFace in your research, please cite our paper:

```bibtex
@article{kang2024oodface,
  title={OODFace: Benchmarking Robustness of Face Recognition under Open-World Corruptions and Variations},
  author={Kang, Caixin and Chen, Yubo and Ruan, Shouwei and Zhao, Shiji and Zhang, Ruochen and Wang, Jiayi and Fu, Shan and Wei, Xingxing},
  journal={arXiv preprint arXiv:2412.02479},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License. Some submodules (e.g., HiSD) are licensed under CC BY-NC-SA 4.0. Please check the LICENSE files in the respective directories.

## 🙏 Acknowledgments

- [BeautyGAN](http://liusi-group.com/projects/BeautyGAN) - Makeup transfer
- [HiSD](https://github.com/imlixinyang/HiSD) - Hierarchical Style Disentanglement
- [PTI](https://github.com/danielroich/PTI) - Pivotal Tuning Inversion
- [imagecorruptions](https://github.com/bethgelab/imagecorruptions) - Image corruption library


## 🔗 Related Links

- [Paper](https://arxiv.org/pdf/2412.02479)
- [Project Homepage](TBD)

---

**Note**: This project is for academic research purposes only. When using commercial APIs, please comply with the respective terms of service.
