# L3-Net
This is the reproduction of Baidu's L3-Net which is used for automaous driving especially in the field of relocalization.
L3_Net is proposed in the paper: Towards Learning based LiDAR Localization for Autonomous Driving, which was published on CVPR.

There are two steps to train the model:
First, we need to run  utils/save_prepared_data.py to preprocess the point clouds and pose data.
Second, we run main/train.py to train the model.
Ps: You need to congif the model by modifying the files in uitls/config/samples/sample_L3Net_dataprepare and in uitls/config/samples/sample_L3Net to let it run in your won computer. Besides, set --cfg_dir="../utils/config/samples/sample_L3Net/" --batch_size=XX in Pycharm's edit configuratoin.

The dataset can be find in https://apollo.auto/southbay.html
Notice that this repository is used only for study, not for commerical use.
