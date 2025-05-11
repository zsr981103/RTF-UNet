# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 14:16:16 2021
网络预测，带IMF
@author: 666
"""
import numpy as np
import torch
from DnCNN import *
from UNet import *
from torch import nn
import torch
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from get_patches import *
###数据和模型载入###
# 数据载入


# path = 'data/noise_mat_npy_data/part1_4_npy_mat/2007BP_part3_11shot.sgy'
# path = 'D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\sgy\Sea_2_10_shot.sgy'
# origin, nSample,extent_time = get_info_seg(path)
# sigma = 1000
# noise = np.random.normal(0, sigma / 255.0, origin.shape)
# noise_data = origin + noise
# noise_data = np.load('data/noise_mat_npy_data/npy/snr_6.npy')
# path = 'data/noise_mat_npy_data/part1_4_npy_mat/2007BP_part1_11shot.sgy'
path = 'data/field_data/Land_0_1_shot.sgy'
noise_data, nSample, extent_time = get_info_seg(path)
# noise_data = np.load('data/noise_mat_npy_data/part1_4_npy_mat/snr_-5.013031825339793.npy')
origin = noise_data
# origin = np.load('data/record_result/sea/DnCNN/denoise_result.npy')
# origin = get_mat('data/record_result/land/VMD_2D/VMD_2D_denoise.mat')

print(calculate_snr(origin,noise_data))
# noise_data = origin
noise_patches, origin_patches = predict_data_extract_paired_patches(noise_data=noise_data,clean_data=origin,patch_length=256,stride=128)

# 模型参数载入
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ResNet18().to(device)
data_in_channel = 1
# model = UNet(in_channels=data_in_channel).to(device)
model = DnCNN().to(device)
weights_path = "D:\Deep\FFTUNet_Project\data_and_result\Shot_normalization\\DnCNN\sigma700\model.pth"
# weights_path = "data/results/model.pth"
# weights_path = "data/Bayesian/run_alpha_1.000_beta_0.054/model.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))

###数据预处理###

# 归一化
def normalization(data, _range):
    return data / _range

# 预测集归一化
range_p = np.max(np.abs(np.concatenate([origin_patches, noise_patches], axis=0)))
p_norm = normalization(noise_patches, range_p)
o_norm = normalization(origin_patches, range_p)
# np.save('range_p', range_p)
#
# 格式转换
# p_norm = noise_patches
# o_norm = origin_patches
p_data = torch.from_numpy(p_norm)
p_data = p_data.type(torch.FloatTensor)
o_data = torch.from_numpy(o_norm)
o_data = o_data.type(torch.FloatTensor)

fft_s = torch.fft.fft(p_data, dim=-1)  # 对最后一个维度做 1D FFT
real_s = fft_s.real
imag_s = fft_s.imag

if data_in_channel == 3:
    fft_input_s = torch.cat([real_s, imag_s, o_data], dim=1)  # 形状: (batch, 3, length)
    p_data = fft_input_s
    print(3)
elif data_in_channel == 2:
    # fft_input_s = torch.cat([p_data,o_data], dim=1)
    fft_input_s = torch.cat([real_s, imag_s], dim=1)
    p_data = fft_input_s
    print(2)
else:
    print(1)
#
###数据去噪###
# 网络预测
train_start_time = time.time()
model.eval()
with torch.no_grad():
    output = model(p_data.to(device))["out"]
    # output = torch.squeeze(output).cpu().detach().numpy()
    output = output.cpu().detach().numpy()

# 数据重排和反归一化
output = output * range_p

# 假设
data_size = 256
stride = 128
useful_start = 64  # 每个片段中使用的起始位置
useful_end = 192   # 每个片段中使用的结束位置
useful_len = useful_end - useful_start  # 中间有效长度128

# 降噪结果重建
# total_batch, _, data_size = output.shape  # 例如 output.shape = (4000, 1, 256)
total_batch, _, _data_size = output.shape  # 例如 output.shape = (4000, 1, 256)
n_samples, n_traces = noise_data.shape
segments_per_trace = total_batch // n_traces  # 每条震道用了多少个batch，应该是5
# num_segments = n_samples // data_size  # 完整的小块数 (4个256)
# remaining_samples = n_samples % data_size  # 如果有剩余部分
# 计算期望长度（用于后续重建）
expected_len = (segments_per_trace - 1) * stride + data_size

# 恢复后的数据将会是 (1151, 800) 形状
# reconstructed_data = np.zeros((n_samples, n_traces))  # 初始化一个空的数组来存放恢复后的震道数据
# 初始化重建数据与叠加次数（用于平均）
reconstructed_data = np.zeros((expected_len, n_traces))
# contribution_count = np.zeros((expected_len, n_traces))
# 遍历每个震道进行重建
# for i in range(n_traces):
#     # 当前震道对应的 batch 范围
#     start_batch = i * segments_per_trace
#     end_batch = (i + 1) * segments_per_trace
#     # 拿出这条震道对应的所有 batch，展平为一个一维数组
#     trace_batches = output[start_batch:end_batch, 0, :]  # 取出这条震道所有batch
#     # trace_full = trace_batches.flatten()  # 将 5 x 256 展开成一条
#     # # 截取前 1151 个点（去掉 padding）
#     # trace_full = trace_full[:n_samples]
#     # # 放入重建后的矩阵
#     # reconstructed_data[:, i] = trace_full
#     for j, segment in enumerate(trace_batches):
#         start_idx = j * stride + useful_start
#         end_idx = j * stride + useful_end
#         if end_idx > expected_len:
#             break  # 防止越界
#         reconstructed_data[start_idx:end_idx, i] += segment[useful_start:useful_end]
#         contribution_count[start_idx:end_idx, i] += 1


for i in range(n_traces):
    start_batch = i * segments_per_trace
    end_batch = (i + 1) * segments_per_trace
    trace_batches = output[start_batch:end_batch, 0, :]  # shape: (segments_per_trace, 256)

    for j, segment in enumerate(trace_batches):
        seg_start = j * stride
        is_first = (j == 0)
        is_last = (j == segments_per_trace - 1)

        if is_first:
            # 第一个patch，使用前192个点
            patch_part = segment[:192]  # 0～192
            write_start = 0
            write_end = 192
        elif is_last:
            # 最后一个patch，使用后192～256的64个点
            patch_part = segment[64:]  # 192～256
            write_start = seg_start + 64
            write_end = seg_start + 256
            if write_end > expected_len:  # 越界保护
                patch_part = patch_part[:expected_len - write_start]
                write_end = expected_len
        else:
            # 中间patch，使用中间部分64～192
            patch_part = segment[64:192]
            write_start = seg_start + 64
            write_end = seg_start + 192

        # 写入重建矩阵（直接替换，无需平均）
        reconstructed_data[write_start:write_end, i] = patch_part


# 裁剪成原始长度（n_samples）
reconstructed_data = reconstructed_data[:n_samples, :]


train_end_time = time.time()
print(train_end_time-train_start_time)
print(calculate_rmse(origin,reconstructed_data))
print(calculate_snr(origin,reconstructed_data))
#
# print(calculate_snr(torch.from_numpy(origin).type(torch.FloatTensor),torch.from_numpy(noise_data).type(torch.FloatTensor)))
# print(calculate_snr(torch.from_numpy(origin).type(torch.FloatTensor),torch.from_numpy(reconstructed_data).type(torch.FloatTensor)))
# plot_seismic_npy(noise_data,extent_time,show=True)
# plot_seismic_npy(origin,extent_time,show=True)
plot_seismic_npy(reconstructed_data,extent_time,show=True)
plot_seismic_npy((noise_data-reconstructed_data),extent_time,show=True)
# plot_seismic_f_k_npy(noise_data,show=True)
# plot_seismic_f_k_npy(origin,show=True)
plot_seismic_f_k_npy(reconstructed_data,show=True)
plot_seismic_f_k_npy((noise_data-reconstructed_data),show=True)
# np.save('denoise_result', reconstructed_data)