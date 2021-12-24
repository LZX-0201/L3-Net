# L3-Net
This is the reproduction of Baidu's L3-Net which is used for automaous driving especially in the field of relocalization. L3_Net is proposed in the paper: *Towards Learning based LiDAR Localization for Autonomous Driving*, which was published on CVPR.
The architecture of the relocalization neural network is as follows:

![image](https://github.com/LZX-0201/L3-Net/blob/main/images/Network%20architecture.png)

## Install
### Install the required packages
```shell
git clone https://github.com/LZX-0201/L3-Net.git
cd L3-Net
pip install -r requirements
```
### Download Apollo-SouthBay Dataset
```shell
https://apollo.auto/southbay.html
```

## Data Pre-processing
Config the data pre-processing process by modifying the configuration file.
```shell
vim uitls/config/samples/sample_L3Net_dataprepare/root_config.yaml
```
Pre-process the training data before training, which selects key-points in the point clouds and uses data from IMU to calculate the ground truth offset of cars pose.
```shell
python utils/save_prepared_data.py
```

## Train
### Configuration
Config the data pre-processing process by modifying the configuration file.
```shell
vim uitls/config/samples/sample_L3Net/root_config.yaml
vim uitls/config/samples/sample_L3Net/dataset/HighWay237.yaml
vim uitls/config/samples/sample_L3Net/model/probability_offset_model.yaml
```
### Train the network
```shell
cd main
python main/train.py --cfg_dir="../utils/config/samples/sample_L3Net/"
```
Ps: It doesn't take too many epoches for the model to converge.
The checkpoints will be saved in L3-Net/checkpoints.

## Test
Test the network using the saved checkpoint.
```shell
cd main
pyton test_L3Net.py --cfg_dir="../utils/config/samples/sample_L3Net/" --check_point_file=../checkpoints/<epoch_num_check_point.pth>
```
The test indicators is same with the paper, which include:

**Horiz. RMS; Horiz. Max; Long. RMS; Lat. RMS; <0.05m Pct. ; <0.6m Pct. ; <0.7m Pct. ; <0.8m Pct. ; Yaw. RMS; Yaw. Max; <0.1° Pct. ; <0.3° Pct. ; <0.6° Pct. **

## Supplement
1. This repository use the previous frame point cloud as the map.
2. This repository doesn't contain the Temporal Smoothness part.
3. This repository is used only for study, not for commercial use.
