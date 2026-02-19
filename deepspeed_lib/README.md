# 为什么 ZeRO-2 显存没降？ZeRO-3 反而更高？

模型参数个数为500M

- **模型权重 (Model Weights, FP16):** 0.5 Billion×2 bytes≈1 GB
- **梯度 (Gradients, FP16):** 反向传播时产生的梯度。 0.5 Billion×2 bytes≈1 GB
- **优化器状态 (Optimizer States, FP32):** **这是显存杀手！** 如果你用的是 AdamW，它需要为每个参数保存两个状态：动量 (Momentum) 和 方差 (Variance)，而且通常保持 FP32 精度以保证收敛。 0.5 Billion×8 bytes(4+4)≈4 GB
- **主权重副本 (Master Weights, FP32):** 混合精度训练通常需要一份 FP32 的权重副本用于更新参数。 0.5 Billion×4 bytes≈2 GB

**仅“静态”部分合计：** 1+1+4+2=8 GB

#### 2. 动态部分 (随 Batch Size 和层数变化)

- **激活值 (Activations):** 为了进行反向传播计算梯度，前向传播（Forward）时的每一层输出都需要保留。 虽然你的 Batch Size 只有 16，但因为你有 **32层** 且宽度为 **4096**，这部分开销也不容小觑。
- **临时缓冲区 (Temporary Buffers):** PyTorch 计算时的临时空间。
- **显存碎片 (Fragmentation):** 申请和释放显存过程中产生的无法使用的碎片空间（通常会浪费 1GB+）。

**总结：** **8GB (静态参数/优化器) + ~4.3GB (激活值/碎片/Buffer) = ~12.3GB** 



zero1  

 GPU 0 | Max Memory: 6163.13 MB 

 GPU 1 | Max Memory: 6163.13 MB 

 zero2  

 GPU 0 | Max Memory: 6163.13 MB 

 GPU 1 | Max Memory: 6163.13 MB 

 zero3 

 GPU 0 | Max Memory: 7117.04 MB 

 GPU 1 | Max Memory: 7117.04 MB 

这是一个非常敏锐的观察！你遇到的现象揭示了 DeepSpeed 在中小规模模型（0.5B）上的 **“固定开销（Overhead）效应”**。

简单来说：**由于模型还不够大，DeepSpeed 为了“加速”而预分配的显存缓冲区（Buffers），超过了它为你“节省”下来的参数显存。**

我们逐一拆解这个“反直觉”现象：

### 1. 为什么 ZeRO-2 和 ZeRO-1 占用完全一样？(6163.13 MB)

* **理论差异**：ZeRO-2 比 ZeRO-1 多切分了梯度（Gradient）。
* **实际情况**：
    * 你的模型梯度总大小约为 **1GB**（FP16）。双卡切分后，理论上每张卡节省 **0.5GB**。
    * **但 DeepSpeed 有一个机制叫 `Gradient Bucketing`（梯度桶）**。
        * 为了防止频繁的小包通信（这会严重拖慢训练速度），DeepSpeed 会预先分配一块连续的大显存（Bucket），把计算出来的梯度填进去，填满后再统一通信。
        * **默认的 Bucket 大小通常是 500MB 到 1GB。**
    * **结论**：DeepSpeed 虽然切分了逻辑上的梯度，但它预分配的这个“通信桶”占用的物理显存，可能刚好填补了节省下来的那部分空间。所以你看到的显存数字在小数点后两位都一样，说明这是**框架层的预分配开销**。

### 2. 为什么 ZeRO-3 显存（7.1 GB）反而比 ZeRO-2（6.1 GB）更高？

这是最经典的问题。ZeRO-3 切分了模型参数，理论上显存应该最低，但为什么反了？

**原因：ZeRO-3 的“预取（Prefetch）”和“重建（Reconstruction）”开销。**

* **参数切分带来的节省**：
    * 模型权重约 **1GB**。双卡切分，每张卡只存 **0.5GB**。
    * **ZeRO-3 帮你省下的显存 = 0.5 GB**。
* **参数重建带来的额外开销**：
    * 在 ZeRO-3 中，当 GPU 0 需要计算第一层时，它手里只有一半参数，它必须从 GPU 1 拉取另一半。
    * 为了不让计算停下来等数据（卡顿），DeepSpeed 会开启 **Prefetch（预取）**：在计算第 N 层时，偷偷把 N+1 层的参数拉到显存里备用。
    * 这就需要分配额外的 **Prefetch Buffer**。
    * 此外，为了合并参数，还需要 **Live Parameter Buffer**。
* **算总账**：
    * 节省：-0.5 GB (权重)
    * 新增开销：+1.5 GB (预取 Buffer + 通信 Buffer + 碎片)
    * **结果：ZeRO-3 反而多占了 1GB 显存！**

---

### 如何让 ZeRO-3 展现真正的实力？（解决方案）

对于 0.5B 这种“小”模型，ZeRO-3 的默认配置太“豪放”了。我们需要 **调小 Buffer**，牺牲一点点通信速度，来换取极致的显存降低。

请创建一个新的配置文件 `config_z3_tiny.yaml`，我们强制限制 DeepSpeed 的胃口：

#### `config_z3_tiny.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
  # === 关键修改：限制 ZeRO-3 的 Buffer 大小 ===
  # 强制调小 Buffer，防止小模型上出现 Overhead 倒挂
  zero_optimization:
    stage3_prefetch_bucket_size: 500000   # 默认是 5e7 (5000万)，这里改为 50万
    stage3_param_persistence_threshold: 10000 # 只要参数超过这个数就切分
    stage3_max_live_parameters: 10000000  # 限制同时存活的完整参数量
    reduce_bucket_size: 500000           # 减小梯度通信桶
    stage3_gather_16bit_weights_on_model_save: true
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

### 再次运行验证

请使用这个经过“瘦身”配置的 ZeRO-3 运行：

```bash
accelerate launch --config_file config_z3_tiny.yaml train_demo_v2.py
```

**预期结果：**
这次 ZeRO-3 的显存占用应该会下降到 **5GB 左右，甚至更低**，终于低于 ZeRO-1/2 了。

**总结一条经验法则：**
* **模型巨大（>10B）时**：ZeRO-3 节省的显存（几十GB）远大于它的 Buffer 开销，优势巨大。
* **模型较小（<1B）时**：ZeRO-3 的 Buffer 开销会吃掉节省的红利，导致出现你看到的“倒挂”现象。此时 **ZeRO-2** 往往是最佳平衡点。