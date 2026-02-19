import torch

def test_p2p_bandwidth():
    # 检查是否有足够的 GPU
    if torch.cuda.device_count() < 2:
        print("当前环境只有单卡或无 GPU，需要至少两张显卡才能测试！")
        return

    print(f"检测到 {torch.cuda.device_count()} 张显卡。正在准备测试...")
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')

    # 准备 1GB 的数据 (256M 个 float32 元素 = 1024 MB)
    size_in_bytes = 1024 * 1024 * 1024
    # 使用 empty 可以瞬间分配显存而不进行多余计算
    tensor0 = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device=device0)

    # 1. 预热 (Warmup) - 让 GPU 预先分配好底层资源，避免冷启动误差
    for _ in range(5):
        tensor1 = tensor0.to(device1)
    torch.cuda.synchronize()

    # 2. 开始测速
    iterations = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录起点
    start_event.record()
    for _ in range(iterations):
        tensor1 = tensor0.to(device1)
    # 记录终点
    end_event.record()

    # 等待所有 CUDA 任务完成
    torch.cuda.synchronize()

    # 3. 计算时间与带宽
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_s = elapsed_time_ms / 1000.0

    total_transferred_gb = (size_in_bytes * iterations) / (1024**3)
    bandwidth = total_transferred_gb / elapsed_time_s

    print("-" * 40)
    print(f"传输总数据量: {total_transferred_gb} GB")
    print(f"总耗时: {elapsed_time_s:.4f} 秒")
    print(f"👉 GPU 0 到 GPU 1 的实际单向带宽估算: **{bandwidth:.2f} GB/s**")
    print("-" * 40)

test_p2p_bandwidth()