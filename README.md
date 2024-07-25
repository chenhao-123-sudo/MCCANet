<div align="center">
  <p>
    <a href="https://ultralytics.com/yolov8" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

[English](README.md) | [简体中文](README.zh-CN.md)
<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

MFRENet is a deep network model specially used to process UAV images. Compared with YOLOv8, MFRENET achieves higher recognition accuracy with little change in parameter number and computation amount.

</a>

<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.tiktok.com/@ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://discord.gg/7aegy5d8" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/blob/main/social/logo-social-discord.png" width="2%" alt="" /></a>
</div>
</div>

## <div align="center">Documentation</div>

See below for a quickstart installation and usage example.

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
pip install ultralytics
pip install timm
pip install einops
```

</details>

<details open>
<summary>Usage</summary>

#### CLI

MFRENet may be used directly in the Command Line Interface (CLI) with a command:

```bash
train:
python main.py
resume:
python resume.py
val:
python val.py
```

</details>

