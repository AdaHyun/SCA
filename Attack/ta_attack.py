"""
Adaptive Template Attack (TA) - robust version
- 自动检测类别数量并在样本少时切换到 HW 模式
- PCA 降噪以提高协方差估计稳定性
- Key-recovery 部分增加索引映射与越界保护，防止索引错误
- 生成与之前兼容的输出文件（混淆矩阵、class-rank、GE/SR 等）
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy.random as npr

# ========== 路径配置 ==========
try:
    from path_config import DATA_PATH, RESULT_PATH
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from path_config import DATA_PATH, RESULT_PATH

DATASET_PATH = os.path.join(DATA_PATH, "ASCAD_mini_8000points.h5")
RESULT_DIR = os.path.join(RESULT_PATH, "TA")
_RAW_DEFAULT = "/root/autodl-tmp/SCA/data/rawDatabase/ASCAD.h5"
RAW_PATH = _RAW_DEFAULT if os.path.exists(_RAW_DEFAULT) else None
BYTE_IDX = 2
N_TRIALS = 5
SEED = 0


# ========== 工具函数 ==========

def load_dataset(path):
    with h5py.File(path, "r") as f:
        Xp = f["Profiling_traces/traces"][:]
        yp = f["Profiling_traces/labels"][:]
        Xa = f["Attack_traces/traces"][:]
        ya = f["Attack_traces/labels"][:]
    return Xp, yp.astype(np.uint8), Xa, ya.astype(np.uint8)


def standardize(X):
    X = X.astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    X /= (X.std(axis=1, keepdims=True) + 1e-9)
    return X


def select_features(X, y, k):
    F, _ = f_classif(X, y)
    idx = np.argsort(-F)[:k]
    return np.sort(idx)


def fit_templates(X, y, classes):
    n_classes = len(classes)
    n_feat = X.shape[1]
    means = np.zeros((n_classes, n_feat))
    covs = np.zeros((n_classes, n_feat, n_feat))
    for i, c in enumerate(classes):
        Xc = X[y == c]
        if Xc.shape[0] < 2:
            means[i] = 0
            covs[i] = np.eye(n_feat)
        else:
            means[i] = Xc.mean(axis=0)
            covs[i] = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(n_feat)
    return means, covs


def gaussian_loglik(X, mean, cov):
    n = X.shape[1]
    cov = cov + 1e-6 * np.eye(n)
    inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    diff = X - mean
    term2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return -0.5 * (n * np.log(2 * np.pi) + logdet + term2)


# ========== 主程序 ==========

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"[TA] Loading dataset: {DATASET_PATH}")
    Xp, yp, Xa, ya = load_dataset(DATASET_PATH)
    Xp, Xa = standardize(Xp), standardize(Xa)

    # 自动判断是否使用 HW 标签（样本少 --> HW）
    unique_labels = np.unique(yp)
    if len(unique_labels) < 50:
        # already compact labels (unlikely), keep as is
        USE_HW = False
    elif len(Xp) < 5000:
        USE_HW = True
    else:
        USE_HW = False

    print(f"[TA] Detected {len(unique_labels)} unique labels, using HW_MODE={USE_HW}")

    if USE_HW:
        yp = np.array([bin(int(v)).count("1") for v in yp], dtype=np.uint8)
        ya = np.array([bin(int(v)).count("1") for v in ya], dtype=np.uint8)

    classes = np.unique(yp)
    n_classes = len(classes)

    # 特征选择 + PCA
    k_feat = min(300, Xp.shape[1])
    feat_idx = select_features(Xp, yp, k_feat)
    Xp_sel = Xp[:, feat_idx]
    Xa_sel = Xa[:, feat_idx]

    if Xp_sel.shape[1] > 150:
        print("[TA] Applying PCA for stability (retain 95% variance)...")
        pca = PCA(n_components=0.95, svd_solver='full')
        Xp_sel = pca.fit_transform(Xp_sel)
        Xa_sel = pca.transform(Xa_sel)
        print(f"    -> reduced to {Xp_sel.shape[1]} components")

    # Fit templates
    print(f"[TA] Fitting {len(classes)} Gaussian templates...")
    means, covs = fit_templates(Xp_sel, yp, classes)

    # Compute log-likelihoods for attack traces
    print("[TA] Computing log-likelihoods...")
    loglik = np.zeros((len(classes), Xa_sel.shape[0]), dtype=np.float64)
    for i, c in enumerate(classes):
        loglik[i] = gaussian_loglik(Xa_sel, means[i], covs[i])

    # simple prediction & accuracy
    y_pred = classes[np.argmax(loglik, axis=0)]
    acc = accuracy_score(ya, y_pred)
    print(f"[TA] Attack accuracy: {acc:.4f}")

    # Confusion (top-16)
    vals, cnts = np.unique(ya, return_counts=True)
    order = np.argsort(-cnts)
    top_vals = vals[order][:16]
    mask = np.isin(ya, top_vals)
    cm = confusion_matrix(ya[mask], y_pred[mask], labels=top_vals)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar()
    plt.title("TA Confusion (Top-16 classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(top_vals)), top_vals, rotation=90)
    plt.yticks(np.arange(len(top_vals)), top_vals)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "ta_confusion_top16.png"), dpi=150)
    plt.close()

    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Attack accuracy: {acc:.6f}\n")
        f.write(f"#features: {Xp_sel.shape[1]}\n")
        f.write(f"n_classes={len(classes)}, USE_HW={USE_HW}\n")

    # Class-rank curve
    cumul = np.cumsum(loglik, axis=1)
    m_values = list(range(10, Xa_sel.shape[0] + 1, max(1, Xa_sel.shape[0] // 50)))
    ranks = []
    for m in m_values:
        score_m = cumul[:, m - 1]
        order_m = np.argsort(-score_m)
        true_cls = ya[m - 1]
        # find index of true class within classes array
        idx_true = np.where(classes == true_cls)[0]
        if idx_true.size == 0:
            ranks.append(np.nan)
            continue
        idx_true = idx_true[0]
        rank = int(np.where(order_m == idx_true)[0][0]) + 1
        ranks.append(rank)

    plt.figure(figsize=(6, 3))
    plt.plot(m_values, ranks, marker="o")
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.xlabel("#attack traces (m)")
    plt.ylabel("Class rank (1=best)")
    plt.title("TA class-rank curve")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "ta_class_rank.png"), dpi=150)
    plt.close()

    # ========== Key recovery metrics (GE/SR/TTD) ==========
    if RAW_PATH and os.path.exists(RAW_PATH):
        try:
            print("[TA] Evaluating key recovery metrics...")
            Np = Xp.shape[0]
            Na = Xa.shape[0]
            with h5py.File(RAW_PATH, "r") as fr:
                pts_full = fr['metadata']['plaintext'][:, BYTE_IDX].astype(np.uint8)
                keys_full = fr['metadata']['key'][:, BYTE_IDX].astype(np.uint8)
            pts = pts_full[Np:Np + Na]
            vals, cnts = np.unique(keys_full, return_counts=True)
            true_key = int(vals[np.argmax(cnts)])

            # full AES sbox (256)
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

            # 构造 m_values 与 ranks_trials
            m_values = list(range(10, Na + 1, max(1, Na // 50)))
            ranks_trials = np.zeros((len(m_values), N_TRIALS), dtype=np.float32)

            n_classes_loglik = loglik.shape[0]
            rng = npr.default_rng(SEED)
            for t in range(N_TRIALS):
                print(f"  Trial {t+1}/{N_TRIALS}")
                order = rng.permutation(Na)
                pts_ord = pts[order]
                ll_all = loglik[:, order]  # shape: (n_classes_loglik, Na)
                for mi, m in enumerate(m_values):
                    scores = np.zeros(256, dtype=np.float64)
                    # 对每个 key guess k，计算其 score
                    for k in range(256):
                        # sbox mapping of plaintext byte ^ k
                        idx = sbox[pts_ord[:m] ^ k]  # shape (m,)
                        # 如果当前在 HW 模式（模板数量<=9 或者我们检测到 USE_HW），把 sbox 值映射为 HW
                        if n_classes_loglik <= 9:
                            # idx 是 uint8 值，转换为 Hamming weight
                            # 使用 unpackbits 高效并行计算
                            bits = np.unpackbits(idx.astype(np.uint8)[:, None], axis=1)
                            hw = bits.sum(axis=1)
                            idx_mapped = hw
                        else:
                            idx_mapped = idx
                        # 安全 clip，防止任何越界
                        idx_mapped = np.clip(idx_mapped, 0, n_classes_loglik - 1).astype(int)
                        # accumulate ll_all[idx_mapped, range(m)]
                        scores[k] = ll_all[idx_mapped, np.arange(m)].sum()
                    order_k = np.argsort(-scores)
                    # 如果 true_key 不在 order_k 中（理论上不可能），会抛错 -> 防护
                    pos = np.where(order_k == true_key)[0]
                    ranks_trials[mi, t] = (pos[0] + 1) if pos.size > 0 else n_classes_loglik

            ge = ranks_trials.mean(axis=1)
            sr = (ranks_trials == 1).mean(axis=1)
            # TTD 列表（可能为空）
            ttd_list = []
            for t in range(N_TRIALS):
                hit = np.where(ranks_trials[:, t] == 1)[0]
                if hit.size > 0:
                    ttd_list.append(int(m_values[hit[0]]))

            # 绘图并保存
            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ge, marker='o')
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('#attack traces (m)')
            plt.ylabel('GE (avg rank)')
            plt.title(f'TA GE curve (byte {BYTE_IDX})')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "ta_ge_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, sr, marker='o')
            plt.xlabel('#attack traces (m)')
            plt.ylabel('Success rate (rank=1)')
            plt.title(f'TA SR curve (byte {BYTE_IDX})')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "ta_sr_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ranks_trials[:, 0], marker='o')
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('#attack traces (m)')
            plt.ylabel('Key rank (1=best)')
            plt.title('TA key-rank curve (trial 0)')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "ta_key_rank.png"), dpi=150)
            plt.close()

            # CSV 输出
            import csv
            with open(os.path.join(RESULT_DIR, "ta_metrics.csv"), "w", newline="") as fw:
                w = csv.writer(fw)
                w.writerow(["m", "GE", "SR"])
                for m, g, s in zip(m_values, ge, sr):
                    w.writerow([int(m), float(g), float(s)])

            # summary 追加
            with open(os.path.join(RESULT_DIR, "summary.txt"), "a", encoding="utf-8") as fw:
                fw.write(f"TA metrics with raw data (byte {BYTE_IDX})\n")
                fw.write(f"GE@{m_values[-1]}={float(ge[-1]):.3f}, SR@{m_values[-1]}={float(sr[-1]):.3f}\n")
                if len(ttd_list) > 0:
                    fw.write(f"TTD(mean/median)={np.mean(ttd_list):.1f}/{np.median(ttd_list):.1f}\n")
                fw.write(f"Recovered={'YES' if (sr[-1] > 0 or len(ttd_list) > 0) else 'NO'}\n")

        except Exception as e:
            print("TA key recovery failed:", e)


if __name__ == "__main__":
    main()
