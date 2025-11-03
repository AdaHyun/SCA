"""
基于 CNN 的侧信道攻击示例 (PyTorch)
------------------------------------------------------------
流程:
1. 使用 Profiling traces 训练 1D CNN，将 trace → 256 类 (SBOX 输出)
2. 在 Attack traces 测试集上评估 Top-1 准确率
3. 保存训练曲线、混淆矩阵 (Top-16 类)
4. 使用网络输出的 softmax likelihood 计算 Key Recovery 指标:
   - GE (Guessing Entropy)
   - SR (Success Rate)
   - TTD (Time to Disclosure)
------------------------------------------------------------
依赖:
  pip install torch torchvision torchaudio scikit-learn matplotlib h5py
"""

import os
import sys
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ======================================
# 路径配置 (适配你的服务器)
# ======================================
DATASET_PATH = "/root/autodl-tmp/SCA/data/rawDatabase/ASCAD_mini_8000points.h5"
RESULT_DIR = "/root/SCA/Result/CNN/"
RAW_PATH = "/root/autodl-tmp/SCA/data/rawDatabase/ASCAD.h5"
BYTE_IDX = 2
N_TRIALS = 5
SEED = 0


# ======================================
# 数据加载与预处理
# ======================================
def load_mini_dataset(path: str):
    with h5py.File(path, "r") as f:
        Xp = f["Profiling_traces/traces"][:]
        yp = f["Profiling_traces/labels"][:]
        Xa = f["Attack_traces/traces"][:]
        ya = f["Attack_traces/labels"][:]
    return Xp.astype(np.float32), yp.astype(np.int64), Xa.astype(np.float32), ya.astype(np.int64)


def standardize_per_trace(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-9
    return (X / std).astype(np.float32)


# ======================================
# CNN 模型定义
# ======================================
class SimpleCNN(nn.Module):
    """简单的 1D CNN"""
    def __init__(self, input_len: int, n_classes: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            out = self.features(dummy)
            feat_dim = out.shape[1] * out.shape[2]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ======================================
# 训练函数
# ======================================
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * yb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.size(0))

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)

        model.eval()
        with torch.no_grad():
            vtotal, vcorrect = 0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                vcorrect += int((pred == yb).sum().item())
                vtotal += int(yb.size(0))
        val_acc = vcorrect / max(1, vtotal)
        hist["val_acc"].append(val_acc)
        print(f"[Epoch {ep}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return hist


# ======================================
# 主程序
# ======================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    Xp, yp, Xa, ya = load_mini_dataset(DATASET_PATH)
    Xp = standardize_per_trace(Xp)
    Xa = standardize_per_trace(Xa)

    Xp_t = torch.from_numpy(Xp[:, None, :])
    yp_t = torch.from_numpy(yp)
    Xa_t = torch.from_numpy(Xa[:, None, :])
    ya_t = torch.from_numpy(ya)

    train_loader = DataLoader(TensorDataset(Xp_t, yp_t), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xa_t, ya_t), batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    model = SimpleCNN(input_len=Xp.shape[1], n_classes=256).to(device)
    start = time.time()
    hist = train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3)
    dur = time.time() - start

    # Attack Top-1 测试
    model.eval()
    with torch.no_grad():
        logits = model(Xa_t.to(device)).cpu().numpy()
    y_pred = logits.argmax(axis=1)
    acc = accuracy_score(ya, y_pred)
    print(f"[CNN] Attack Top-1 Accuracy: {acc:.4f} (time {dur:.1f}s)")

    # ================= 图表输出 =================
    plt.figure(figsize=(8, 3))
    plt.plot(hist["train_loss"], label="train_loss")
    plt.plot(hist["train_acc"], label="train_acc")
    plt.plot(hist["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("CNN training curves")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "cnn_train_curves.png"), dpi=150)
    plt.close()

    vals, cnts = np.unique(ya, return_counts=True)
    order = np.argsort(-cnts)
    top_vals = vals[order][:16]
    cm = confusion_matrix(ya, y_pred, labels=top_vals)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("CNN Confusion (Top-16 classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(top_vals)), top_vals, rotation=90)
    plt.yticks(np.arange(len(top_vals)), top_vals)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "cnn_confusion_top16.png"), dpi=150)
    plt.close()

    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as fw:
        fw.write(f"Attack Top-1 acc: {acc:.6f}\n")
        fw.write(f"Train time: {dur:.1f}s\n")

    # ================= Class-rank =================
    with torch.no_grad():
        lp = torch.log_softmax(model(Xa_t.to(device)), dim=1).cpu().numpy()  # (N, 256)
    cumul = np.cumsum(lp, axis=0)
    m_values = list(range(5, Xa.shape[0] + 1, max(1, Xa.shape[0] // 50)))
    ranks = []
    for m in m_values:
        score_m = cumul[m - 1]
        order = np.argsort(-score_m)
        true_cls = int(ya[m - 1])
        ranks.append(int(np.where(order == true_cls)[0][0]) + 1)
    plt.figure(figsize=(6, 3))
    plt.plot(m_values, ranks, marker="o")
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.xlabel("#attack traces (m)")
    plt.ylabel("Class rank (1=best)")
    plt.title("CNN class-rank curve")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "cnn_class_rank.png"), dpi=150)
    plt.close()

    return model, Xp, yp, Xa, ya, lp


# ======================================
# Key Recovery Evaluation
# ======================================
if __name__ == "__main__":
    model, Xp, yp, Xa, ya, lp = main()

    if RAW_PATH and os.path.exists(RAW_PATH):
        try:
            print("[CNN] Evaluating key recovery metrics...")
            lp = lp.T  # -> (256, Na)
            Np = Xp.shape[0]
            Na = Xa.shape[0]
            with h5py.File(RAW_PATH, "r") as fr:
                pts_full = fr["metadata"]["plaintext"][:, BYTE_IDX].astype(np.uint8)
                keys_full = fr["metadata"]["key"][:, BYTE_IDX].astype(np.uint8)
            pts = pts_full[Np:Np + Na]
            vals, cnts = np.unique(keys_full, return_counts=True)
            true_key = int(vals[np.argmax(cnts)])

            sbox = np.array([
                0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
                0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
                0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
                0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
                0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
                0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
                0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
                0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
                0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
                0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
                0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
                0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
                0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
                0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
                0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
                0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
            ], dtype=np.uint8)

            rng = npr.default_rng(SEED)
            m_values = list(range(10, Na + 1, max(1, Na // 50)))
            ranks_trials = np.zeros((len(m_values), N_TRIALS), dtype=np.float32)

            for t in range(N_TRIALS):
                order = rng.permutation(Na)
                lp_all = lp[:, order]
                pts_ord = pts[order]
                for mi, m in enumerate(m_values):
                    scores = np.zeros(256, dtype=np.float64)
                    for k in range(256):
                        v_idx = sbox[pts_ord[:m] ^ k]
                        scores[k] = float(lp_all[v_idx, np.arange(m)].sum())
                    order_k = np.argsort(-scores)
                    ranks_trials[mi, t] = int(np.where(order_k == true_key)[0][0]) + 1

            ge = ranks_trials.mean(axis=1)
            sr = (ranks_trials == 1).mean(axis=1)
            ttd_list = [int(m_values[np.where(ranks_trials[:, t] == 1)[0][0]]) for t in range(N_TRIALS) if np.any(ranks_trials[:, t] == 1)]

            # 绘图
            os.makedirs(RESULT_DIR, exist_ok=True)
            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ge, marker="o")
            plt.yscale("log")
            plt.gca().invert_yaxis()
            plt.xlabel("#attack traces (m)")
            plt.ylabel("GE (avg rank)")
            plt.title(f"CNN GE curve (byte {BYTE_IDX})")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "cnn_ge_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, sr, marker="o")
            plt.xlabel("#attack traces (m)")
            plt.ylabel("Success rate (rank=1)")
            plt.title(f"CNN SR curve (byte {BYTE_IDX})")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "cnn_sr_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ranks_trials[:, 0], marker="o")
            plt.yscale("log")
            plt.gca().invert_yaxis()
            plt.xlabel("#attack traces (m)")
            plt.ylabel("Key rank (1=best)")
            plt.title(f"CNN key-rank curve (trial 0)")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "cnn_key_rank.png"), dpi=150)
            plt.close()

            valid_ttd = [v for v in ttd_list if v is not None]
            if len(valid_ttd) > 0:
                plt.figure(figsize=(5, 3))
                plt.hist(valid_ttd, bins=min(10, len(valid_ttd)))
                plt.xlabel("TTD (first m with rank=1)")
                plt.ylabel("count")
                plt.title(f"CNN TTD histogram (byte {BYTE_IDX})")
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, "cnn_ttd_hist.png"), dpi=150)
                plt.close()

            import csv
            with open(os.path.join(RESULT_DIR, "cnn_metrics.csv"), "w", newline="") as fw:
                w = csv.writer(fw)
                w.writerow(["m", "GE", "SR"])
                for m, g, s in zip(m_values, ge, sr):
                    w.writerow([int(m), float(g), float(s)])

            with open(os.path.join(RESULT_DIR, "summary.txt"), "a", encoding="utf-8") as fw:
                fw.write(f"CNN key-recovery metrics (byte {BYTE_IDX})\n")
                fw.write(f"GE@{m_values[-1]}={float(ge[-1]):.3f}, SR@{m_values[-1]}={float(sr[-1]):.3f}\n")
                if len(valid_ttd) > 0:
                    fw.write(f"TTD(mean/median)={np.mean(valid_ttd):.1f}/{np.median(valid_ttd):.1f}\n")
                fw.write(f"Recovered={'YES' if (sr[-1] > 0 or len(valid_ttd) > 0) else 'NO'}\n")

            print("✅ CNN Key Recovery Evaluation Completed!")

        except Exception as e:
            print("❌ CNN 多指标评测失败:", e)
