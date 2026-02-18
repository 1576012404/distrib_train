import torch
import torch.nn as nn
from accelerate import Accelerator

# 模拟数据
x = torch.randn(64, 1024).cuda()
y = torch.randn(64, 10).cuda()
model = nn.Linear(1024, 10).cuda()
optimizer = torch.optim.Adam(model.parameters())

# --- 方案 A: 原生 PyTorch 完整 AMP (必须这么写才安全) ---
print("运行原生 AMP...")
# 必须手动创建这个缩放器
scaler = torch.cuda.amp.GradScaler()

for _ in range(5):
    optimizer.zero_grad()
    # 1. 开启自动混合精度上下文
    with torch.cuda.amp.autocast():
        output = model(x)
        loss = nn.MSELoss()(output, y)

    # 2. 缩放损失并反向传播（防止梯度下溢）
    scaler.scale(loss).backward()

    # 3. scaler.step 会先将梯度除以缩放因子，如果梯度不是 Inf/NaN，则执行 optimizer.step()
    scaler.step(optimizer)

    # 4. 更新缩放因子
    scaler.update()

# --- 方案 B: Accelerate 做法 ---
print("运行 Accelerate...")
accelerator = Accelerator(mixed_precision="fp16")
model, optimizer = accelerator.prepare(model, optimizer)

for _ in range(5):
    optimizer.zero_grad()
    # 自动处理了 autocast
    output = model(x)
    loss = nn.MSELoss()(output, y)

    # 自动处理了梯度缩放(Scaling)
    accelerator.backward(loss)
    optimizer.step()