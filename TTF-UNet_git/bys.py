import argparse
import torch.nn.functional as F
import datetime
from sklearn.model_selection import train_test_split
from skopt.space import Integer, Categorical
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
# UNet
from torch.utils.data import Subset
from UNet import UNet
# DnCNN
from get_patches import *
# 导入贝叶斯优化相关库
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

#
# 定义优化的超参数空间
space = [
    Real(7e-1, 1, prior='log-uniform', name='alpha'),
    Real(1e-2, 2e-1, name='beta'),

]


@use_named_args(space)
def objective(alpha, beta):
    print(f"alpha: {alpha:.3f},beta: {beta:.3f}")
    title = (f"alpha_{alpha:.3f}_beta_{beta:.3f}")

    # if stride >= patch:
    #     # 如果 stride 不合法，则返回一个很差的 loss（惩罚）
    #     return 1e6
    # model selection
    # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 1
    net = UNet(in_channels=input_channels).to(device)
    ###超参数设置###

    EPOCH = 150  # 遍历数据集次数 100
    BATCH_SIZE_s = 100  # 批处理尺寸(batch_size)
    BATCH_SIZE_v = 100
    LR = 5e-5  # 0.000012
    rate = 0.90  # 学习率衰变
    iteration = 20  # 每10次衰减

    ###数据读取###
    data_path = 'data/2007BP_synthetic_train.sgy'
    data, seismic_time, time_length = get_info_seg(data_path)
    clean_data = data
    sigma = 700
    np.random.seed(42)  # 固定随机种子，42 可以换成任意整数
    noise = np.random.normal(0, sigma / 255.0, clean_data.shape)
    noise_data = clean_data + noise
    print(calculate_snr(data, noise_data))

    noise_data, clean_data = extract_paired_patches(clean_data=clean_data, noise_data=noise_data, patch_length=256,
                                                    stride=128)
    x_train, x_val, y_train, y_val = train_test_split(
        clean_data, noise_data, test_size=0.2, random_state=42
    )

    # 截取数据为BATCH倍数
    train_len = (len(x_train) // BATCH_SIZE_s) * BATCH_SIZE_s
    val_len = (len(x_val) // BATCH_SIZE_v) * BATCH_SIZE_v
    x_train = x_train[:train_len]
    y_train = y_train[:train_len]
    x_val = x_val[:val_len]
    y_val = y_val[:val_len]
    print(f"总样本数: {len(noise_data)}")
    print(f"训练集: {len(x_train)}, 验证集: {len(x_val)}")
    Ls = len(x_train)
    Lv = len(x_val)

    ###数据预处理###
    # 归一化
    def normalization(data, _range):
        return data / _range

    # 训练集归一化

    range_s = np.max(np.abs(np.concatenate([x_train, y_train], axis=0)))
    x_t = normalization(x_train, range_s)
    y_t = normalization(y_train, range_s)


    # 验证集归一化
    range_v = np.max(np.abs(np.concatenate([x_val, y_val], axis=0)))
    x_v = normalization(x_val, range_v)
    y_v = normalization(y_val, range_v)


    # 训练集格式转换
    x1_s = torch.from_numpy(x_t)
    x2_s = torch.from_numpy(y_t)
    x1_s = x1_s.type(torch.FloatTensor)
    x2_s = x2_s.type(torch.FloatTensor)
    # 训练集：x2_s 的傅里叶变换
    fft_s = torch.fft.fft(x2_s, dim=-1)  # 对最后一个维度做 1D FFT
    real_s = fft_s.real
    imag_s = fft_s.imag
    fft_input_s = torch.cat([real_s, imag_s, x1_s], dim=1)  # 形状: (batch, 3, length)

    # 验证集格式转换
    x1_v = torch.from_numpy(x_v)
    x2_v = torch.from_numpy(y_v)
    x1_v = x1_v.type(torch.FloatTensor)
    x2_v = x2_v.type(torch.FloatTensor)
    # 验证集：x2_v 的傅里叶变换
    fft_v = torch.fft.fft(x2_v, dim=-1)
    real_v = fft_v.real
    imag_v = fft_v.imag
    fft_input_v = torch.cat([real_v, imag_v, x1_v], dim=1)  # 形状: (batch, 3, length)
    if input_channels == 3:
        x2_s = fft_input_s
        x2_v = fft_input_v

    # 数据封装打乱顺序
    train_data = TensorDataset(x2_s, x1_s)
    val_data = TensorDataset(x2_v, x1_v)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_s, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE_v, shuffle=False, num_workers=0, drop_last=True)

    ###网络训练###
    # 定义是否使用GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型定义
    # net = DnCNN(channels=1).to(device)
    # net = SCWNet18().to(device)

    # criterion = nn.MSELoss()
    criterion = nn.MSELoss(reduction='sum', size_average=False)
    criterion.cuda()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iteration, gamma=rate)
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # 开始训练

    # 生成唯一的文件夹名
    results_dir = os.path.join('./data/Bayesian', f'run_{title}')
    # results_dir = 'data/results'
    os.makedirs(results_dir, exist_ok=True)

    Losslist_s = []
    Losslist_v = []
    best_loss = 100
    # save_path = './net.pth'
    print("Start Training!")
    train_start_time = time.time()
    for epoch in range(EPOCH):
        # print('\nEpoch: %d' % (epoch + 1))
        # if epoch % iteration == 19:
        #     LR = LR * rate
        scheduler.step()

        loss_s = 0.0
        loss_v = 0.0

        snr_after_list = []
        snr_val_after_list = []


        start_time = time.time()
        # for i in range(Ls//BATCH_SIZE_s):
        for i, data_s in enumerate(train_loader, 0):
            net.train()
            net.zero_grad()
            optimizer.zero_grad()
            input_s, target_s = data_s
            input_s = input_s.to(device)
            # target_s = target_s.to(input_s.device)
            # output_s = net(input_s)* target_s
            output = net(input_s)
            output_s = output["out"]
            output_s_f = output["feature"]
            clean_feature = F.adaptive_avg_pool1d(target_s, 16).to(device)
            target_s = target_s.to(device)
            loss_s0 = criterion(output_s, target_s)
            loss_s1 = criterion(output_s_f, clean_feature)
            loss_s_total = alpha * loss_s0 + beta * loss_s1
            loss_s_total.backward()
            # loss_s0.backward()
            optimizer.step()
            # loss_s += loss_s0.item()
            loss_s += loss_s_total.item()
            snr_denoise = calculate_snr(target_s, output_s)
            snr_after_list.append(snr_denoise)

        net.eval()
        with torch.no_grad():
            # for j in range(Lv//BATCH_SIZE_v):
            for j, data_v in enumerate(val_loader, 0):
                input_v, target_v = data_v
                input_v = input_v.to(device)
                # target_v = target_v.to(input_v.device)
                # output_v = net(input_v)*target_v    #前向算法
                # output_v = net(input_v)["out"]  # 前向算法
                output = net(input_v)
                output_v = output["out"]
                output_v_f = output["feature"]
                clean_feature = F.adaptive_avg_pool1d(target_v, 16).to(device)
                target_v = target_v.to(device)
                loss_v0 = criterion(output_v, target_v)
                loss_v1 = criterion(output_v_f, clean_feature)
                loss_v_total = alpha * loss_v0 + beta * loss_v1
                # target_v = target_v.to(device)
                # loss_v0 = criterion(output_v, target_v)
                # loss_v += loss_v0.item()
                loss_v += loss_v_total.item()
                snr_denoise = calculate_snr(target_v, output_v)
                snr_val_after_list.append(snr_denoise)

            if loss_v_total < best_loss:
                best_loss = loss_v_total
                model_name = f'model.pth'
                torch.save(net.state_dict(), os.path.join(results_dir, model_name))

        epoch_avg_train_loss = loss_s / (Ls // BATCH_SIZE_s)
        epoch_avg_snr_after = sum(snr_after_list) / len(snr_after_list)
        epoch_avg_val_loss = loss_v / (Lv // BATCH_SIZE_s)
        epoch_avg_val_snr_after = sum(snr_val_after_list) / len(snr_val_after_list)
        elapsed_time = time.time() - start_time
        Losslist_s.append(epoch_avg_train_loss)
        Losslist_v.append(epoch_avg_val_loss)


        # ===== 打印 epoch 信息 =====
        # print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {epoch_avg_train_loss:.10f}")
        print(f"_Val_ Loss: {epoch_avg_val_loss:.10f}")
        print(f"Train SNR  After: {epoch_avg_snr_after:.4f}")
        print(f"_Val_ SNR  After: {epoch_avg_val_snr_after:.4f}")
        print('Epoch %d, Time: %2.3f' % (epoch + 1, elapsed_time))
        # ===== 写入 txt 文件（仅数字） =====
        with open(os.path.join(results_dir, 'train_loss.txt'), 'a') as f:
            f.write(f"{epoch_avg_train_loss:.8f}\n")
        with open(os.path.join(results_dir, 'val_loss.txt'), 'a') as f:
            f.write(f"{epoch_avg_val_loss:.8f}\n")
        with open(os.path.join(results_dir, 'snr_train_after.txt'), 'a') as f:
            f.write(f"{epoch_avg_snr_after:.8f}\n")
        with open(os.path.join(results_dir, 'snr_val_after.txt'), 'a') as f:
            f.write(f"{epoch_avg_val_snr_after:.8f}\n")


    print('finished training')
    train_end_time = time.time()
    elapsed_seconds = int(train_end_time - train_start_time)
    with open(os.path.join(results_dir, 'time.txt'), 'a') as f:
        f.write(f"{elapsed_seconds:.8f}\n")


    return float(best_loss)


result = gp_minimize(objective, space, n_calls=15, random_state=150)

