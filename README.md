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

