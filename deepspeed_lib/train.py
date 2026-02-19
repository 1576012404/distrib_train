import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
import time


# 1. 定义一个“占显存”的假模型
# 为了让T4显卡(16GB)能跑起来且能看到差异，我们定义一个中等规模的模型
class HeavyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义几个大矩阵，模拟 GPT 类模型的参数量
        # 4096 * 4096 * 4 bytes * 10 layers approx = 670MB parameters
        # 加上梯度和优化器状态，显存会迅速上升
        self.layers = nn.ModuleList([
            nn.Linear(4096, 4096) for _ in range(12)
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


# 2. 定义假数据集
class RandomDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 4096)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    # 初始化 Accelerator
    accelerator = Accelerator()

    # 强制同步打印，防止日志混乱
    def print_rank0(msg):
        if accelerator.is_main_process:
            print(f"[Main Process] {msg}")

    print_rank0(">>> 初始化模型和数据...")

    model = HeavyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = RandomDataset()
    dataloader = DataLoader(dataset, batch_size=16)  # Batch size 较小以聚焦模型本身的显存

    # 关键步骤：使用 accelerator 接管对象
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    print_rank0(">>> 开始训练 Step...")

    model.train()
    start_time = time.time()

    # 运行几个 step 即可
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(batch)
        loss = output.mean()
        accelerator.backward(loss)
        optimizer.step()

        if i >= 5:  # 跑 5 个 batch 就够了
            break

    end_time = time.time()

    # 统计显存
    # 等待所有进程到达此处
    accelerator.wait_for_everyone()

    # 获取当前 GPU 的显存峰值 (MB)
    mem_usage = torch.cuda.max_memory_allocated() / 1024 / 1024

    # 打印结果
    print(f"GPU {accelerator.process_index} | Max Memory Allocated: {mem_usage:.2f} MB")

    if accelerator.is_main_process:
        print_rank0(f"Time taken for 5 steps: {end_time - start_time:.4f}s")
        print_rank0(">>> 实验结束")


if __name__ == "__main__":
    main()