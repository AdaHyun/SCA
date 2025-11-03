"""
MIA（互信息分析）示例脚本（基于已给定的“中间值标签”做泄漏检测）

思路：
- 将标签 label（SBOX(p[2]^k[2])）作为离散类变量；
- 对每个时间点 t 的波形 X[:, t] 与标签 y 计算互信息 MI(X_t; y)；
- 绘制 MI 曲线，观察峰值位置（表示该时间点对类别的判别信息更强）。

数据文件：D:/Study/VScodeProject/SCA/DATA_analyasis/GithubGenerate/ResultDatabased/ASCAD_mini.h5
可选原始文件（含 plaintext/key，用于严格 key-rank）：
  例如 D:/Study/VScodeProject/Failed_SideChannelAttack/sca-benchmark/data/ASCAD.h5
输出目录：D:/Study/VScodeProject/SCA/Result/MIA/
"""
import os
import sys
try:
    from path_config import DATA_PATH, RESULT_PATH
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from path_config import DATA_PATH, RESULT_PATH
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import numpy.random as npr

# 统一路径与评测参数
DATASET_PATH = os.path.join(DATA_PATH, "ASCAD_mini_8000points.h5")
RESULT_DIR = os.path.join(RESULT_PATH, "MIA")
_RAW_DEFAULT = "/root/autodl-tmp/SCA/data/rawDatabase/ASCAD.h5"
RAW_PATH = _RAW_DEFAULT if os.path.exists(_RAW_DEFAULT) else None
BYTE_IDX = 2
N_TRIALS = 5
SEED = 0


# 使用统一的路径配置
DATASET_PATH = os.path.join(DATA_PATH, "ASCAD_mini_8000points.h5")
RESULT_DIR = os.path.join(RESULT_PATH, "MIA")
RAW_PATH = None  # 可选原始 ASCAD.h5，用于严格 key-rank
BYTE_IDX = 2


def load_mini_dataset(path: str):
    with h5py.File(path, "r") as f:
        Xp = f["Profiling_traces/traces"][:]
        yp = f["Profiling_traces/labels"][:]
        Xa = f["Attack_traces/traces"][:]
        ya = f["Attack_traces/labels"][:]
    return Xp, yp.astype(np.uint8), Xa, ya.astype(np.uint8)


def standardize_per_trace(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-9
    X = X / std
    return X


def compute_mi_profile(X: np.ndarray, y: np.ndarray, max_points: int | None = None) -> np.ndarray:
    """计算每个时间点与标签的互信息。
    - X: (N, T)
    - y: (N,), 离散标签（0..255）
    - max_points: 若不为 None，则下采样时间点以加速（例如 2 表示每 2 点取 1 点）
    返回：长度 T 的 MI 向量（若下采样会再插值回 T）。
    """
    N, T = X.shape
    # sklearn.mutual_info_classif 一次可计算多特征的 MI，输入形状 (N, n_features)
    # 这里直接对所有时间点计算，若 T 很大可考虑下采样或分块
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    return mi


def compute_mi_histogram(X: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """直方图法估计 MI：将每个时间点的连续值分箱，计算 I(X_binned; y)。
    简单稳妥，样本 500–2000 即可看到趋势。
    """
    N, T = X.shape
    y_vals = np.unique(y)
    mi = np.zeros(T, dtype=np.float64)
    for t in range(T):
        xt = X[:, t]
        # 分箱（等频或等宽；这里用等宽）
        hist, edges = np.histogram(xt, bins=n_bins)
        bins = np.digitize(xt, edges[:-1], right=True)
        # 估计联合与边缘分布
        px = np.bincount(bins, minlength=n_bins+1).astype(np.float64)
        px = px / px.sum()
        py = np.bincount(y, minlength=int(y_vals.max())+1).astype(np.float64)
        py = py / py.sum()
        pxy = np.zeros((n_bins+1, int(y_vals.max())+1), dtype=np.float64)
        for i in range(N):
            pxy[bins[i], y[i]] += 1.0
        pxy /= N
        # 互信息：sum_{x,y} pxy * log(pxy/(px*py))
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = pxy / (px[:, None] * py[None, :]+1e-12)
            term = pxy * np.log(frac + 1e-12)
        mi[t] = np.nansum(term)
    return mi


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    Xp, yp, Xa, ya = load_mini_dataset(DATASET_PATH)
    # 互信息用 profiling 或 attack 都可，这里用 profiling（更像“建模/分析阶段”）
    Xp = standardize_per_trace(Xp)

    # 同时给出两种 MI：sklearn 与直方图法
    mi = compute_mi_profile(Xp, yp)
    mi_hist = compute_mi_histogram(Xp, yp, n_bins=20)

    # 绘图
    plt.figure(figsize=(10, 3))
    plt.plot(mi)
    plt.title("MIA mutual information profile (Profiling set)")
    plt.xlabel("Time index")
    plt.ylabel("MI")
    plt.tight_layout()
    out_png = os.path.join(RESULT_DIR, "mia_profile.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # 直方图法曲线
    plt.figure(figsize=(10, 3))
    plt.plot(mi_hist)
    plt.title("MIA histogram-based MI (Profiling set)")
    plt.xlabel("Time index")
    plt.ylabel("MI (hist)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "mia_profile_hist.png"), dpi=150)
    plt.close()

    top_t = int(np.argmax(mi))
    print("[MIA] 最高 MI 位置：", top_t, "MI=", float(mi[top_t]))
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as fw:
        fw.write(f"最高 MI 位置（sklearn）：{top_t}, MI={float(mi[top_t]):.6f}\n")
        fw.write(f"最高 MI 位置（hist）  ：{int(np.argmax(mi_hist))}, MI={float(mi_hist.max()):.6f}\n")

    # 若可用原始 raw（含 plaintext/key），做严格 key-rank（以 MI 得分作为评分）
    if RAW_PATH and os.path.exists(RAW_PATH):
        try:
            Np = Xp.shape[0]
            Na = Xa.shape[0]
            with h5py.File(RAW_PATH, "r") as fr:
                pts = fr['metadata']['plaintext'][Np:Np+Na, BYTE_IDX].astype(np.uint8)
                keys = fr['metadata']['key'][:Np+Na, BYTE_IDX].astype(np.uint8)
                vals, cnts = np.unique(keys, return_counts=True)
                true_key = int(vals[np.argmax(cnts)])

            X = Xa
            X = standardize_per_trace(X)
            Na, T = X.shape
            K = 256
            m_values = list(range(10, Na+1, max(1, Na//50)))
            ranks = []
            for m in m_values:
                Xm = X[:m, :]
                scores = np.zeros(K, dtype=np.float64)
                for kguess in range(K):
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
                    Hk = np.array([bin(int(sbox[p ^ kguess])).count("1") for p in pts[:m]], dtype=np.int32)
                    mi_k = mutual_info_classif(Xm, Hk, discrete_features=False, random_state=0)
                    scores[kguess] = float(mi_k.max()) if mi_k.size > 0 else 0.0
                order = np.argsort(-scores)
                r = int(np.where(order == true_key)[0][0]) + 1 if true_key in order else None
                ranks.append(r)

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ranks, marker='o')
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('#attack traces (m)')
            plt.ylabel('Key rank (1=best)')
            plt.title('MIA key-rank curve (byte %d)' % BYTE_IDX)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "mia_key_rank.png"), dpi=150)
            plt.close()
        except Exception as e:
            print("严格 MIA 失败：", e)


if __name__ == "__main__":
    main()
    # 追加：若可用 raw 数据，基于 MIA 得分做多指标评测（GE/SR/TTD）
    if RAW_PATH and os.path.exists(RAW_PATH):
        try:
            Xp, yp, Xa, ya = load_mini_dataset(DATASET_PATH)
            Xp = standardize_per_trace(Xp)
            Xa = standardize_per_trace(Xa)
            Np = Xp.shape[0]
            Na = Xa.shape[0]
            with h5py.File(RAW_PATH, "r") as fr:
                pts_full = fr['metadata']['plaintext'][:, BYTE_IDX].astype(np.uint8)
                keys_full = fr['metadata']['key'][:, BYTE_IDX].astype(np.uint8)
            pts = pts_full[Np:Np+Na]
            vals, cnts = np.unique(keys_full, return_counts=True)
            true_key = int(vals[np.argmax(cnts)])

            rng = npr.default_rng(SEED)
            m_values = list(range(10, Na+1, max(1, Na//50)))
            K = 256
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

            ranks_trials = np.zeros((len(m_values), N_TRIALS), dtype=np.float32)
            for t in range(N_TRIALS):
                order = rng.permutation(Na)
                Xm_all = Xa[order]
                pts_ord = pts[order]
                for mi, m in enumerate(m_values):
                    Xm = Xm_all[:m, :]
                    scores = np.zeros(K, dtype=np.float64)
                    for kguess in range(K):
                        Hk = np.array([bin(int(sbox[p ^ kguess])).count("1") for p in pts_ord[:m]], dtype=np.int32)
                        mi_k = mutual_info_classif(Xm, Hk, discrete_features=False, random_state=0)
                        scores[kguess] = float(mi_k.max()) if mi_k.size > 0 else 0.0
                    order_k = np.argsort(-scores)
                    ranks_trials[mi, t] = int(np.where(order_k == true_key)[0][0]) + 1

            ge = ranks_trials.mean(axis=1)
            sr = (ranks_trials == 1).mean(axis=1)
            ttd_list = []
            for t in range(N_TRIALS):
                hit = np.where(ranks_trials[:, t] == 1)[0]
                ttd_list.append(int(m_values[hit[0]]) if hit.size > 0 else None)

            os.makedirs(RESULT_DIR, exist_ok=True)
            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ge, marker='o')
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('#attack traces (m)')
            plt.ylabel('GE (avg rank)')
            plt.title('MIA GE curve (byte %d)' % BYTE_IDX)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "mia_ge_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, sr, marker='o')
            plt.xlabel('#attack traces (m)')
            plt.ylabel('Success rate (rank=1)')
            plt.title('MIA SR curve (byte %d)' % BYTE_IDX)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "mia_sr_curve.png"), dpi=150)
            plt.close()

            valid_ttd = [v for v in ttd_list if v is not None]
            if len(valid_ttd) > 0:
                plt.figure(figsize=(5, 3))
                plt.hist(valid_ttd, bins=min(10, len(valid_ttd)))
                plt.xlabel('TTD (first m with rank=1)')
                plt.ylabel('count')
                plt.title('MIA TTD histogram (byte %d)' % BYTE_IDX)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, "mia_ttd_hist.png"), dpi=150)
                plt.close()

            import csv
            with open(os.path.join(RESULT_DIR, "mia_metrics.csv"), 'w', newline='') as fw:
                w = csv.writer(fw)
                w.writerow(["m", "GE", "SR"])
                for m, g, s in zip(m_values, ge, sr):
                    w.writerow([int(m), float(g), float(s)])

            with open(os.path.join(RESULT_DIR, "summary.txt"), "a", encoding="utf-8") as fw:
                fw.write(f"MIA metrics with raw data (byte {BYTE_IDX})\n")
                fw.write(f"GE@{m_values[-1]}={float(ge[-1]):.3f}, SR@{m_values[-1]}={float(sr[-1]):.3f}\n")
                if len(valid_ttd) > 0:
                    fw.write(f"TTD(mean/median)={np.mean(valid_ttd):.1f}/{np.median(valid_ttd):.1f}\n")
                fw.write(f"Recovered={'YES' if (sr[-1] > 0 or len(valid_ttd) > 0) else 'NO'}\n")
        except Exception as e:
            print("MIA 多指标评测失败：", e)
