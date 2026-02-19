import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
import time


# === V2改动：大幅增加模型深度，压榨显存 ===
class HeavyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 显存杀手：
        # 之前的 12 层 -> 现在的 32 层
        # 参数量翻了近 3 倍，梯度和优化器状态也会翻 3 倍
        # 预计 DDP 模式下显存会飙升到 13GB - 15GB 甚至 OOM
        self.layers = nn.ModuleList([
            nn.Linear(4096, 4096) for _ in range(32)
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            # 使用 checkpointing (可选) 这里故意不用，为了撑爆显存
            x = self.activation(layer(x))
        return x


class RandomDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 4096)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    accelerator = Accelerator()

    def print_rank0(msg):
        if accelerator.is_main_process:
            print(f"[Main Process] {msg}")

    print_rank0(">>> [V2-压力测试] 初始化 32层 HeavyModel...")

    model = HeavyModel()
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"Total Parameters: {total_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = RandomDataset()
    # Batch size 保持 16，主要靠模型参数撑显存
    dataloader = DataLoader(dataset, batch_size=16)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    print_rank0(">>> 开始训练 Step...")
    model.train()
    start_time = time.time()

    try:
        # 跑 10 个 batch，让显存稳定下来
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(batch)
            loss = output.mean()
            accelerator.backward(loss)
            optimizer.step()

            if i % 2 == 0:
                print_rank0(f"Step {i} finished")

            if i >= 10:
                break
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"!!! GPU {accelerator.process_index} OOM (爆显存了) !!!")
        else:
            raise e

    end_time = time.time()

    accelerator.wait_for_everyone()
    mem_usage = torch.cuda.max_memory_allocated() / 1024 / 1024/1024
    print(f"GPU {accelerator.process_index} | Max Memory: {mem_usage:.2f} GB")

    if accelerator.is_main_process:
        print_rank0(f"Time: {end_time - start_time:.4f}s")
        print_rank0(">>> 实验结束")


if __name__ == "__main__":
    main()