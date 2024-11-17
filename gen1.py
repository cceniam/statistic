import torch
import numpy as np
from scipy.stats import ttest_ind

# 设置固定的随机种子以确保结果的可复现性
np.random.seed(42)

def generate_data_worker(data1, data2, t_value, max_iterations):
    learning_rate = 0.001
    noise_scale = 0.05

    # 将数据移动到GPU上
    data1 = torch.tensor(data1, device=device, dtype=torch.float64)
    data2 = torch.nn.Parameter(torch.tensor(data2, device=device, dtype=torch.float64))

    optimizer = torch.optim.AdamW([data2], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)

    for iteration in range(max_iterations):
        # 计算当前的t值（保持计算在GPU上）
        mean1 = torch.mean(data1)
        mean2 = torch.mean(data2)
        var1 = torch.var(data1, unbiased=False)
        var2 = torch.var(data2, unbiased=False)
        n1 = data1.size(0)
        n2 = data2.size(0)

        current_t = (mean1 - mean2) / torch.sqrt(var1 / n1 + var2 / n2)

        # 提前停止条件更严格
        if torch.abs(current_t - t_value) < 0.001:
            print(f"在第 {iteration} 次迭代时提前停止，当前t值: {current_t.item()}")
            # 进入微调阶段
            fine_tune_lr = learning_rate * 0.001
            fine_tune_optimizer = torch.optim.AdamW([data2], lr=fine_tune_lr)
            for _ in range(500):
                current_t = (mean1 - mean2) / torch.sqrt(var1 / n1 + var2 / n2)
                loss = torch.pow(t_value - current_t, 2)
                fine_tune_optimizer.zero_grad()
                loss.backward()
                fine_tune_optimizer.step()

            # 将结果离散化为整数
            with torch.no_grad():
                data2 = torch.round(data2)
            return data2.detach().cpu().numpy()

        # 计算损失（使用平方差损失并增加敏感性）
        loss = torch.pow(t_value - current_t, 4)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([data2], max_norm=2.0)
        optimizer.step()
        scheduler.step()

        # 添加噪声（仅在前50%迭代中）
        if iteration < max_iterations // 2:
            with torch.no_grad():
                random_noise = torch.normal(0, noise_scale, size=data2.shape, device=device) * (
                        max_iterations - iteration) / max_iterations
                data2.add_(random_noise)

        noise_scale = max(0.001, noise_scale * 0.99)

    # 最终将结果离散化为整数
    with torch.no_grad():
        data2 = torch.round(data2)

    return data2.detach().cpu().numpy()

def generate_data(size, mean1, mean2, std1, std2, t_value, max_iterations=10000):
    # 初始随机数据生成
    data1 = np.random.normal(mean1, std1, size)
    data2 = np.random.normal(mean2, std2 * 1.5, size)  # 增加初始标准差以获得更好的多样性

    # 将数据集 1 离散化为整数
    data1 = np.round(data1)

    adjusted_data2 = generate_data_worker(data1, data2, t_value, max_iterations)

    return data1, adjusted_data2

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 示例用法
if __name__ == "__main__":
    data1, data2 = generate_data(size=109, mean1=21.43, mean2=16.43, std1=3.15, std2=2.91, t_value=7.928)
    print("数据集 1:", data1)
    print("数据集 2:", data2)

    # 验证生成的t值
    t_stat, p_value = ttest_ind(data1, data2)
    print("生成的t值:", t_stat)
