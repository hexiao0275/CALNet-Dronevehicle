
# Multispectral Object Detection via Cross-Modal Conflict-Aware Learning

This repo is the official implementation for **Multispectral Object Detection via Cross-Modal Conflict-Aware Learning**. The paper has been accepted to **ACM MM 2023 Oral**.

### News
[2024.01.10] **Code** released!

[2023.08.18] Our paper is ready!

## Introduction

**Abstract.** Multispectral object detection has gained significant attention due to its potential in all-weather applications, particularly those involving visible (RGB) and infrared (IR) images. Despite substantial advancements in this domain, current methodologies primarily rely on rudimentary accumulation operations to combine complementary information from disparate modalities, overlooking the semantic conflicts that arise from the intrinsic heterogeneity among modalities. To address this issue, we propose a novel learning network, the Cross-modal Conflict-Aware Learning Network (CALNet), that takes into account semantic conflicts and complementary information within multi-modal input. Our network comprises two pivotal modules: the Cross-Modal Conflict Rectification Module (CCR) and the Selected Cross-modal Fusion (SCF) Module. The CCR module mitigates modal heterogeneity by examining contextual information of analogous pixels, thus alleviating multi-modal information with semantic conflicts. Subsequently, semantically coherent information is supplied to the SCF module, which fuses multi-modal features by assessing intra-modal importance to select semantically rich features and mining inter-modal complementary information. To assess the effectiveness of our proposed method, we develop a two-stream one-stage detector based on CALNet for multispectral object detection. Comprehensive experimental outcomes demonstrate that our approach considerably outperforms existing methods in resolving the cross-modal semantic conflict issue and achieving state-of-the-art accuracy in detection results.

## Dataset
### DroneVehicle

The dataset is available for download at the following link. Many heartfelt thanks for their meticulous dataset collection and expert labeling efforts!

[DroneVehicle]([https://github.com/ultralytics/yolov5](https://github.com/VisDrone/DroneVehicle))

Processed labels for our carefully organized dataset:

-[Label]  [download](https://pan.baidu.com/s/17tLn0D6yZkVqokMBpis1jw) password:5sow

Pre-training weights download:

-[Weight]  [download](https://pan.baidu.com/s/1PnmdKqIxPnTgK6yQ6WfwpA) password:zvi2

### CALNet Overview
<div align="left">
<img src="https://github.com/hexiao-cs/CALNet-Dronevehicle/blob/main/img_readme/ccr_scf.png" width="800">
</div>

### Visualization of Detection

<div align="left">
<img src="https://github.com/hexiao-cs/CALNet-Dronevehicle/blob/main/img_readme/showtime.png" width="800">
</div>

# INSTAllation 


I have tested the following versions of OS and softwares：
* OS：Ubuntu 18.04
* CUDA: 11.3

## Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda remove -n mmdetection --all
conda activate mmdet
conda install pytorch==1.10.1 cudatoolkit==11.3.1 torchvision==0.11.2 -c pytorch
```

```
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```
## train

```
python train.py

```
## Install DOTA_devkit. 

### Download DOTA_devkit. 

-[DOTA_devkit]  [download](https://pan.baidu.com/s/1MBW3DK6Vjx09T5dJdiXnig) password:peoe

**(Custom Install, it's just a tool to split the high resolution image and evaluation the obb)**
```
cd yolov5_obb/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## test

```
python valtest.py --save-json --name 'obb_demo6'
python tools/TestJson2VocClassTxt.py --json_path 'runs/val/obb_demo/best_obb_predictions.json' --save_path 'runs/val/obb_demo/obb_predictions_Txt'
python DOTA_devkit-master/dota_evaluation_task1.py 
```


## Acknowledgment

Our codes are mainly based on [yolov5](https://github.com/ultralytics/yolov5). Many thanks to the authors!

## Citation

If this is useful for your research, please consider cite.

```
@inproceedings{he2023multispectral,
  title={Multispectral Object Detection via Cross-Modal Conflict-Aware Learning},
  author={He, Xiao and Tang, Chang and Zou, Xin and Zhang, Wei},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={1465--1474},
  year={2023}
}

@article{he2023object,
  title={Object Detection in Hyperspectral Image via Unified Spectral-Spatial Feature Aggregation},
  author={He, Xiao and Tang, Chang and Liu, Xinwang and Zhang, Wei and Sun, Kun and Xu, Jiangfeng},
  journal={arXiv preprint arXiv:2306.08370},
  year={2023}
}
```


