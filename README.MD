# Content
This repository contains the official implementation of "A Geometric Knowledge Oriented Single-Frame 2D-to-3D Human Absolute Pose Estimation Method".

- [Installation](#installation)
- [Training](#prepare-environment)
- [Acknowledgement](#acknowledgement)

## Installation
### Requirements

- Linux (Ubuntu 18.04.6 LTS) 
- Python 3.8
- PyTorch 1.10.1
- CUDA 9.1+
- requirements.txt

### Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n GKONet python=3.8 -y
conda activate GKONet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
E.g., PyTorch 1.10.1:
```shell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**Important:** Make sure that your compilation CUDA version and runtime CUDA version match.

c. Install other requirements

```shell
pip install -r requirements.txt
```

## Dataset setup
Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./data' directory). 
to download the Human 3.6M dataset (
<code>
data_3d_h36m.npz 
</code>
,
<code>
data_2d_h36m_gt
</code>
and 
<code>
data_2d_h36m_cpn_ft_h36m_dbb.npz
</code>
).

Or you can directly download the processed data in 
[P-STMO](https://github.com/paTRICK-swk/P-STMO).

**Note:** Please put above datasets in the `'./data'` directory.

```bash
${YOUR_ROOT}/
|-- data
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Pretrained model & Test
The pretrained model can be found in [here](https://drive.google.com/drive/folders/1d2_tS51c4mcv_cimCT-2mJCaLIbdBMz8?usp=sharing), please download it and put it in the `'./pretrained'` directory. 

To test pretrained model with detected 2D poses (CPN) on Human3.6M:

```bash
python run_h36m.py --eval True --checkpoint pretrained/H36M_CPN_Depth8_p1_50_43.pth --depth 8 -g 0 -exp TEST
```

To test pretrained model with ground truth poses on Human3.6M:

```bash
python run_h36m.py --eval True --checkpoint pretrained/H36M_GT_Depth6_p1_33_98.pth --depth 6 -g 0 -exp TEST -k gt
```

## Training Model
To train model with detected 2D poses (CPN) on Human3.6M:

```bash
python run_h36m.py -k cpn_ft_h36m_dbb --depth 8 -pd 32 -jd 128 -exp CPN
```

To train model with ground truth poses on Human3.6M:

```bash
python run_h36m.py -k gt --depth 6 -pd 32 -jd 128 -exp GT
```

## Citation
If you find our work useful in your research, please consider citing:

    @ARTICLE{hu2023,
      author={Hu, Mengxian and Liu, Chengju and Li, Shu and Yan, Qingqing and Fang, Qin and Chen, Qijun},
      journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
      title={A Geometric Knowledge Oriented Single-Frame 2D-to-3D Human Absolute Pose Estimation Method}, 
      year={2023},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TCSVT.2023.3279291}}

## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [JointFormer](https://github.com/seblutz/JointFormer)
- [3DMPP](https://github.com/3dpose/3D-Multi-Person-Pose)


## Licence

This project is licensed under the terms of the MIT license.

<div align="left">

