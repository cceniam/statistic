import torch
import numpy as np
from scipy.stats import ttest_ind, truncnorm

# 设置固定的随机种子以确保结果的可复现性
np.random.seed(42)


def generate_data_worker(data1, data2, t_value, std_target, max_iterations):
    learning_rate = 0.001
    noise_scale = 0.05

    # 将数据移动到GPU上
    data1 = torch.tensor(data1, device=device, dtype=torch.float64)
    data2 = torch.nn.Parameter(torch.tensor(data2, device=device, dtype=torch.float64))

    optimizer = torch.optim.AdamW([data2], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)

    min_data1_value = torch.min(data1)  # 获取数据集1的最小值

    for iteration in range(max_iterations):
        # 计算当前的t值（保持计算在GPU上）
        mean1 = torch.mean(data1)
        mean2 = torch.mean(data2)
        var1 = torch.var(data1, unbiased=False)
        var2 = torch.var(data2, unbiased=False)
        n1 = data1.size(0)
        n2 = data2.size(0)

        current_t = (mean1 - mean2) / torch.sqrt(var1 / n1 + var2 / n2)

        # 计算当前的标准差
        current_std = torch.sqrt(var2)

        # 提前停止条件更严格
        if torch.abs(current_t - t_value) < 0.001 and torch.abs(current_std - std_target) < 0.001:
            print(f"在第 {iteration} 次迭代时提前停止，当前t值: {current_t.item()}, 当前标准差: {current_std.item()}")
            # 进入微调阶段
            fine_tune_lr = learning_rate * 0.001
            fine_tune_optimizer = torch.optim.AdamW([data2], lr=fine_tune_lr)
            for _ in range(500):
                current_t = (mean1 - mean2) / torch.sqrt(var1 / n1 + var2 / n2)
                current_std = torch.sqrt(var2)
                loss = torch.pow(t_value - current_t, 2) + torch.pow(std_target - current_std, 2)
                fine_tune_optimizer.zero_grad()
                loss.backward()
                fine_tune_optimizer.step()

                # 确保数据非负且不小于数据集1的最小值
                with torch.no_grad():
                    data2.clamp_(min=min_data1_value)

            # 将结果离散化为整数
            with torch.no_grad():
                data2 = torch.round(data2)
            return data2.detach().cpu().numpy()

        # 计算损失（使用平方差损失并增加敏感性）
        loss = torch.pow(t_value - current_t, 4) + torch.pow(std_target - current_std, 2)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # 保留计算图，以便下一次迭代使用
        torch.nn.utils.clip_grad_norm_([data2], max_norm=2.0)
        optimizer.step()
        scheduler.step()

        # 确保数据非负且不小于数据集1的最小值
        with torch.no_grad():
            data2.clamp_(min=min_data1_value)

        # 添加器序器的随机震荡（仅在前50%迭代中）
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


def generate_data(size1, size2, mean1, mean2, std1, std2, t_value, max_iterations=10000):
    # 使用截断正态分布生成非负的数据集1
    lower, upper = 2, np.inf  # 设置数据的下限为2，上限为正无穷大，确保生成的值为非负
    data1 = truncnorm.rvs((lower - mean1) / std1, (upper - mean1) / std1, loc=mean1, scale=std1, size=size1)

    # 使用截断正态分布生成非负的数据集2
    lower, upper = 0, np.inf  # 设置数据的下限为0，上限为正无穷大，确保生成的值为非负
    data2 = truncnorm.rvs((lower - mean2) / std2, (upper - mean2) / std2, loc=mean2, scale=std2, size=size2)

    # 将数据集 1 和 2 离散化为整数
    data1 = np.round(data1)
    data2 = np.round(data2)

    adjusted_data2 = generate_data_worker(data1, data2, t_value, std2, max_iterations)

    return data1, adjusted_data2


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

if __name__ == "__main__":
    data1, data2 = generate_data(size1=50, size2=59, mean1=33.64, mean2=35.23, std1=6.48, std2=8.21, t_value=1.281)
    print("数据集 1:", data1)
    print("数据集 2:", data2)

    # 验证生成的t值
    t_stat, p_value = ttest_ind(data1, data2)
    print("p值:", p_value)
    print("生成的t值:", t_stat)
    print("生成的标准差1:", np.std(data1))
    print("生成的标准差2:", np.std(data2))
