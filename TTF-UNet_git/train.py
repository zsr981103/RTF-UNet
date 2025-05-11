from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from UNet import *
from DnCNN import *
from get_patches import *


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_channels = 3
net = UNet(in_channels=input_channels).to(device)
# net = DnCNN().to(device)
###超参数设置###

EPOCH = 200  # 遍历数据集次数 100
BATCH_SIZE_s = 100  # 批处理尺寸(batch_size)
BATCH_SIZE_v = 100
LR = 5e-5  # 0.000012
rate = 0.8  # 学习率衰变
iteration = 20  # 每10次衰减
###真实数据读取###
use_real_data = False  # ← 控制是否加入真实数据

### 真实数据读取 ###
if use_real_data:
    data_sea_path = 'data/field_data/Sea_2_10_shot.sgy'
    data_sea,seismic_sea_time,time_sea_length = get_info_seg(data_sea_path)
    noise_data_sea = data_sea
    # data_clean_sea_path = 'data/record_result/sea/VMD_2D/Sea_2_10_shot/VMD_2D_denoise.mat'
    # clean_data_sea = get_mat(data_clean_sea_path)
    data_clean_sea_path = 'data/record_result/sea/DnCNN/Sea_2_10_shot/denoise_result.npy'
    clean_data_sea = np.load(data_clean_sea_path)
    noise_data_sea, clean_data_sea = extract_paired_patches(clean_data=clean_data_sea,noise_data=noise_data_sea,patch_length=256,stride=128)
    print(f"[海洋数据] clean: {clean_data_sea.shape}, noise: {noise_data_sea.shape}")

    data_land_path = 'data/field_data/Land_2_6_shot.sgy'
    data_land,seismic_land_time,time_land_length = get_info_seg(data_land_path)
    noise_data_land = data_land
    data_clean_land_path = 'data/record_result/land/VMD_2D/Land_2_6_shot/VMD_2D_denoise.mat'
    clean_data_land = get_mat(data_clean_land_path)
    noise_data_land, clean_data_land = extract_paired_patches(clean_data=clean_data_land,noise_data=noise_data_land,patch_length=256,stride=128)
    print(f"[陆地数据] clean: {clean_data_land.shape}, noise: {noise_data_land.shape}")



###数据读取###
data_path = 'data/2007BP_synthetic_train.sgy'
data,seismic_time,time_length = get_info_seg(data_path)
clean_data = data

# 固定随机种子
np.random.seed(42)
sigma = 700
noise = np.random.normal(0, sigma / 255.0, clean_data.shape)
noise_data = clean_data + noise
print(calculate_snr(data,noise_data))
noise_data, clean_data = extract_paired_patches(clean_data=clean_data,noise_data=noise_data,patch_length=256,stride=128)


if use_real_data:
    # 合并三个数据集
    clean_data = np.concatenate([clean_data, clean_data_sea, clean_data_land], axis=0)
    noise_data = np.concatenate([noise_data, noise_data_sea, noise_data_land], axis=0)



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
# x_t,y_t,x_v,y_v = x_train,y_train,x_val,y_val

# 训练集格式转换
x1_s = torch.from_numpy(x_t)
x2_s = torch.from_numpy(y_t)
x1_s = x1_s.type(torch.FloatTensor)
x2_s = x2_s.type(torch.FloatTensor)
# 训练集：x2_s 的傅里叶变换
fft_s = torch.fft.fft(x2_s, dim=-1)  # 对最后一个维度做 1D FFT
real_s = fft_s.real
imag_s = fft_s.imag



# 验证集格式转换
x1_v = torch.from_numpy(x_v)
x2_v = torch.from_numpy(y_v)
x1_v = x1_v.type(torch.FloatTensor)
x2_v = x2_v.type(torch.FloatTensor)
# 验证集：x2_v 的傅里叶变换
fft_v = torch.fft.fft(x2_v, dim=-1)
real_v = fft_v.real
imag_v = fft_v.imag


if input_channels == 3:
    fft_input_s = torch.cat([real_s, imag_s, x1_s], dim=1)  # 形状: (batch, 3, length)
    fft_input_v = torch.cat([real_v, imag_v, x1_v], dim=1)  # 形状: (batch, 3, length)
    print("3")
    x2_s = fft_input_s
    x2_v = fft_input_v
elif input_channels == 2:
    print("2")
    fft_input_s = torch.cat([real_s, imag_s], dim=1)  # 形状: (batch, 3, length)
    fft_input_v = torch.cat([real_v, imag_v], dim=1)  # 形状: (batch, 3, length)
    x2_s = fft_input_s
    x2_v = fft_input_v
else:
    print("1")


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


criterion = nn.MSELoss()
criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iteration, gamma=rate)


# 开始训练
# 记录信噪比
snrs_x_n = []
snrs_x_p = []


results_dir = 'data/results'
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
    loss_s = 0.0
    loss_v = 0.0

    snr_before_list = []
    snr_after_list = []
    snr_val_before_list = []
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
        output_s = net(input_s)["out"]
        target_s = target_s.to(device)
        loss_s0 = criterion(output_s, target_s)
        loss_s0.backward()
        optimizer.step()
        loss_s += loss_s0.item()

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
            output_v = net(input_v)["out"]  # 前向算法
            target_v = target_v.to(device)
            loss_v0 = criterion(output_v, target_v)
            loss_v += loss_v0.item()

            snr_denoise = calculate_snr(target_v,output_v)
            snr_val_after_list.append(snr_denoise)
            # 保存最优模型
            if loss_v0 < best_loss:
                best_loss = loss_v0
                model_name = f'model.pth'
                torch.save(net.state_dict(), os.path.join(results_dir, model_name))

    epoch_avg_train_loss = loss_s / (Ls//BATCH_SIZE_s)
    epoch_avg_snr_after = sum(snr_after_list) / len(snr_after_list)
    epoch_avg_val_loss = loss_v / (Lv//BATCH_SIZE_s)
    epoch_avg_val_snr_after = sum(snr_val_after_list) / len(snr_val_after_list)
    elapsed_time = time.time() - start_time
    Losslist_v.append(epoch_avg_val_loss)
    Losslist_s.append(epoch_avg_train_loss)

    # 学习率衰减
    scheduler.step()  # 这一行会更新学习率

    # ===== 打印 epoch 信息 =====
    # print(f"Epoch {epoch + 1} Summary:")
    print(f"Train Loss: {epoch_avg_train_loss:.10f}")
    print(f"_Val_ Loss: {epoch_avg_val_loss:.10f}")
    print(f"Train SNR  After: {epoch_avg_snr_after:.4f}")
    print(f"_Val_ SNR  After: {epoch_avg_val_snr_after:.4f}")
    print('Epoch %d, Time: %3.3f' % (epoch + 1, elapsed_time))
    # ===== 写入 txt 文件（仅数字） =====
    with open(os.path.join(results_dir, 'train_loss.txt'), 'a') as f:
        f.write(f"{epoch_avg_train_loss:.10f}\n")
    with open(os.path.join(results_dir, 'val_loss.txt'), 'a') as f:
        f.write(f"{epoch_avg_val_loss:.10f}\n")
    with open(os.path.join(results_dir, 'snr_train_after.txt'), 'a') as f:
        f.write(f"{epoch_avg_snr_after:.4f}\n")
    with open(os.path.join(results_dir, 'snr_val_after.txt'), 'a') as f:
        f.write(f"{epoch_avg_val_snr_after:.4f}\n")


print('finished training')
train_end_time = time.time()
elapsed_seconds = int(train_end_time - train_start_time)
with open(os.path.join(results_dir, 'time.txt'), 'a') as f:
    f.write(f"{elapsed_seconds:.8f}\n")
###绘图###
# 格式转换
input_v = input_v.cpu()
input_v = input_v.detach().numpy()
output_v = output_v.cpu()
output_v = output_v.detach().numpy()
target_v = target_v.cpu()
target_v = target_v.detach().numpy()

# Loss变化
x = range(1, EPOCH + 1)
y_s = Losslist_s
y_v = Losslist_v
plt.semilogy(x, y_s, 'b.-')
plt.semilogy(x, y_v, 'r.-')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.show()
# plt.savefig("accuracy_loss.jpg")

# 去噪前后绘图
col = 85  # 显示第几个数据去噪效果
imf = 6
x = range(0, len(target_v[0, 0, :]))
y1 = target_v[col, 0, :]
y2 = np.sum(input_v[col, :, :], axis=0)
y3 = output_v[col, 0, :]
plt.plot(x, y1, 'b.-')
plt.plot(x, y2, 'r.-')
plt.plot(x, y3, 'g.-')
plt.xlabel('Time')
plt.ylabel('Ampulitude')
plt.show()

###SNR###
# 去噪前
origSignal = target_v[:, 0, :]
errorSignal = target_v[:, 0, :] - np.sum(input_v, axis=1)
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues1 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues1)

# 去噪后
origSignal = target_v[:, 0, :]
errorSignal = target_v[:, 0, :] - output_v[:, 0, :]
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues2 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues2)

end = time.time()

