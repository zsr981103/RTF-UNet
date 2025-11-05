# RTF-UNet
RTF-UNet code
RTF-UNet
This repository contains the implementation of RTF-UNet, a time–frequency domain collaborative learning model for seismic signal denoising.
RTF-UNet supports flexible configuration for time-domain, frequency-domain, or time–frequency-domain feature learning, and allows joint training on synthetic and real field seismic data (marine and land).
在train.py文件中


train.py — Network Training Configuration


Network Parameters：
input_channels：Controls the type of input features:

1: time-domain features

2: frequency-domain features

3: joint time–frequency features

EPOCH：Number of training epochs.

BATCH_SIZE_s：Batch size for the training set.

BATCH_SIZE_v：Batch size for the validation set.

LR：Initial learning rate.

rate：Learning rate decay factor.

iteration：Number of iterations after which the learning rate decays.


Data Loading Parameters：

use_real_dat：Whether to include real field seismic data in the training set.

data_path：Path to synthetic seismic data.

data_sea_path, data_land_path：Paths to field seismic data (marine and land).

data_clean_sea_path, data_clean_land_path：Paths to the denoised (clean) field seismic data.


Output and Model Saving：
results_dir：Directory for saving all output files.
model_name：Name used when saving the trained network.




denoise_3.py — Denoising Inference

Main Parameters
path：Path to the noisy seismic data (in .sgy format).
origin：Path to the pseudo-label (clean reference) seismic data (in .sgy format).

When using RTF-UNet, set both path and origin.
For other networks, you may set path = origin.

weights_path：Path to the trained model weights.
data_in_channel：Controls the type of input features during inference:
1: time-domain features
2: frequency-domain features
3: time–frequency-domain features

Patch Extraction
predict_data_extract_paired_patches
Function for dividing the data into short segments (patches):
noise_patches, origin_patches = predict_data_extract_paired_patches(
    noise_data=noise_data,
    clean_data=origin,
    patch_length=256,
    stride=128
)
patch_length: number of sample points per patch (e.g., 256).
stride: step size between consecutive patches (e.g., 128).

Visualization
plot_seismic_npy：Visualize denoised seismic results.
plot_seismic_f_k_npy：Visualize the F–K (frequency–wavenumber) spectrum of seismic data.


在网络配置参数方面
input_channels：控制输入通道数
input_channels=1，则是使用时域特征实现网络建模
input_channels=2，则是使用频域特征实现网络建模
input_channels=3，则是使用时频域特征实现网络建模
EPOCH：训练的次数
BATCH_SIZE_s：训练集一批的训练数量
BATCH_SIZE_v：验证集一批的训练数量
LR：初始学习率
rate：学习衰变率
iteration：每训练多少次进行衰变


在数据读取参数中
use_real_dat：是否使用真实地震数据加入到训练集中
data_sea_path与data_land_path：field seismic data数据路径
data_clean_sea_path与data_clean_land_path：经过降噪的field seismic data数据路径
data_path 表示人工合成地震数据路径

在存储文件中
results_dir：表示所有文件的存储路径
model_name：表示网络的存储名称

在denoise_3.py文件中
path：表示待降噪的地震数据路径（.sgy格式）
origin：表示伪标签的地震数据路径（.sgy格式）（当使用RTF-UNet时使用，其他网络可令path=origin）
noise_patches, origin_patches = predict_data_extract_paired_patches(noise_data=noise_data,clean_data=origin,patch_length=256,stride=128)
predict_data_extract_paired_patches：将数据分割成256-simple points
patch_length:序列采样点大小
stride：采样间隔多少采样点
weights_path ：加载存储的网络
data_in_channel：控制输入通道数
data_in_channel=1，则是使用时域特征实现网络建模
data_in_channel=2，则是使用频域特征实现网络建模
data_in_channel=3，则是使用时频域特征实现网络建模

plot_seismic_npy:绘制降噪结果
plot_seismic_f_k_npy：绘制f-k频谱图
