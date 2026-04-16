"""
getminloss.py — 扫描 detector checkpoint，汇总 epoch / test_loss，并标出最优模型。

输入
  - 目录 checkpoint_dir（默认 ./checkpoints）内，文件名形如：
    gpu0_<epoch>_<test_loss>_net_detector.pth

输出
  - 写入 log_file（默认 log.txt）：全表 + 文末「最优模型」摘要（epoch、test_loss、filename）
  - 终端打印保存路径与最优 checkpoint

运行：在 USIP-master 根目录（或含 checkpoints 的路径）执行 python getminloss.py；按需改 checkpoint_dir、log_file。
"""
import os
import re

checkpoint_dir = "./checkpoints"
log_file = "log.txt"

data = []

# 遍历所有 .pth 文件
for filename in os.listdir(checkpoint_dir):
    if filename.endswith(".pth"):
        # 匹配你的文件名格式：gpu0_27_-6.379618_net_detector.pth
        match = re.search(r"gpu0_(\d+)_(-?\d+\.\d+)_net_detector\.pth", filename)
        if match:
            epoch = int(match.group(1))
            test_loss = float(match.group(2))
            data.append({
                "epoch": epoch,
                "test_loss": test_loss,
                "filename": filename
            })

# 按 epoch 排序
data.sort(key=lambda x: x["epoch"])

# 找出 loss 最小（最好）的模型
best_model = min(data, key=lambda x: x["test_loss"])

# 写入 log.txt
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"{'epoch':<6} {'test_loss':<12} filename\n")
    f.write("-" * 60 + "\n")
    for item in data:
        f.write(f"{item['epoch']:<6} {item['test_loss']:<12} {item['filename']}\n")

    # 写入最优模型
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"✅ 最优模型（loss 最小）：\n")
    f.write(f"epoch: {best_model['epoch']}\n")
    f.write(f"test_loss: {best_model['test_loss']}\n")
    f.write(f"filename: {best_model['filename']}\n")

print(f"✅ 信息已保存到 {log_file}")
print(f"🏆 最优模型：{best_model['filename']} (loss: {best_model['test_loss']})")
