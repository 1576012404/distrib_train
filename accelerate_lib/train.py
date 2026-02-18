import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
import time


# --- 1. 定义一个简单的合成数据集 ---
# 目的：排除硬盘 IO 瓶颈，纯测 GPU 速度
class SyntheticDataset(Dataset):
    def __init__(self, size=10000, img_shape=(3, 224, 224)):
        self.size = size
        self.img_shape = img_shape
        # 预先生成数据放在内存里
        self.data = torch.randn(size, *img_shape)
        self.target = torch.randint(0, 1000, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def main():
    # --- 2. 初始化 Accelerator ---
    # 这里会自动检测命令行参数，决定是用 fp16 还是 fp32，是用 1 张卡还是 2 张卡
    accelerator = Accelerator()

    # 打印当前的运行状态
    accelerator.print(f"🚀 启动配置: 进程数={accelerator.num_processes}, "
                      f"精度={accelerator.mixed_precision}, "
                      f"设备={accelerator.device}")

    # --- 3. 准备模型和数据 ---
    # 使用 ResNet50 (不加载预训练权重，只测计算量)
    from torchvision.models import resnet50
    model = resnet50(num_classes=1000)

    # 注意：Batch Size 也是随着卡数线性增加的
    # 单卡 batch=64，双卡实际上 global_batch=128
    batch_size = 64
    dataset = SyntheticDataset(size=2000)  # 跑 2000 个样本做测试
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Prepare (Accelerate 魔法时刻) ---
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # --- 5. 训练循环与计时 ---
    model.train()

    # 预热 (Warmup)：跑几步让 CUDA kernel 初始化，避免影响计时
    accelerator.print("🔥 开始预热...")
    for i, batch in enumerate(dataloader):
        if i > 5: break
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    accelerator.wait_for_everyone()  # 等待所有卡预热完毕

    # 正式计时
    accelerator.print("⏱️  开始正式测试...")
    start_time = time.time()
    total_samples = 0

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()

        # 统计样本数 (注意：如果是多卡，每张卡处理 batch_size 个)
        total_samples += inputs.size(0)

    # 等待所有进程跑完
    accelerator.wait_for_everyone()
    end_time = time.time()

    # --- 6. 结果汇总 ---
    # 只在主进程计算总吞吐量
    if accelerator.is_main_process:
        # 在多卡模式下，total_samples 只是主进程看到的数量
        # 总处理量 = 单进程处理量 * 进程数 (假设数据分配均匀)
        # 或者更严谨的做法是 gather 所有卡的数据，但这里做近似估算即可
        total_processed = total_samples * accelerator.num_processes
        duration = end_time - start_time
        throughput = total_processed / duration

        print("-" * 40)
        print(f"✅ 测试完成！")
        print(f"耗时: {duration:.2f} 秒")
        print(f"总吞吐量: {throughput:.2f} samples/sec")
        print(f"说明: 数值越大，性能越好")
        print("-" * 40)


if __name__ == "__main__":
    main()