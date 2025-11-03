"""
CPA（相关功耗分析）示例脚本 + 密钥恢复指标（GE/SR/TTD）
优化版：向量化 + 进度打印 + 稳定输出
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
import numpy.random as npr

# 路径与评测参数
DATASET_PATH = os.path.join(DATA_PATH, "ASCAD_mini_8000points.h5")
RESULT_DIR = os.path.join(RESULT_PATH, "CPA")
_RAW_DEFAULT = "/root/autodl-tmp/SCA/data/rawDatabase/ASCAD.h5"
RAW_PATH = _RAW_DEFAULT if os.path.exists(_RAW_DEFAULT) else None
BYTE_IDX = 2
N_TRIALS = 5
SEED = 0
KEY_BATCH_SIZE = 64  # 如果显存或内存不足，可改小为32/16


def hw_vec(arr: np.ndarray) -> np.ndarray:
    return np.array([bin(int(x)).count("1") for x in arr], dtype=np.int32)


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
    return X / std


def cpa_leakage_profile(X: np.ndarray, y_labels: np.ndarray, use_hw: bool = True) -> np.ndarray:
    y = hw_vec(y_labels) if use_hw else y_labels.astype(np.int32)
    y = y - y.mean()
    y_std = y.std() + 1e-9
    Xc = X - X.mean(axis=0, keepdims=True)
    Xstd = Xc.std(axis=0) + 1e-9
    cov = (Xc * (y / y_std)[:, None]).mean(axis=0)
    corr = cov / Xstd
    return np.abs(corr)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    Xp, yp, Xa, ya = load_mini_dataset(DATASET_PATH)
    Xp = standardize_per_trace(Xp)
    Xa = standardize_per_trace(Xa)

    corr_hw = cpa_leakage_profile(Xa, ya, use_hw=True)
    corr_raw = cpa_leakage_profile(Xa, ya, use_hw=False)

    plt.figure(figsize=(10, 3))
    plt.plot(corr_hw, label="|corr| with HW(label)")
    plt.plot(corr_raw, label="|corr| with label")
    plt.title("CPA-style leakage profile (Attack set)")
    plt.xlabel("Time index")
    plt.ylabel("|corr|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "cpa_leakage_profile.png"), dpi=150)
    plt.close()

    top_t = int(np.argmax(corr_hw))
    print("[CPA] Top corr (HW) @", top_t, "=", float(corr_hw[top_t]))
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as fw:
        fw.write(f"最高相关位置（HW）：{top_t}, corr={float(corr_hw[top_t]):.6f}\n")

    # ===========================================================
    # 密钥恢复指标（需要 raw 数据）
    # ===========================================================
    if RAW_PATH and os.path.exists(RAW_PATH):
        try:
            Np = Xp.shape[0]
            Na = Xa.shape[0]
            with h5py.File(RAW_PATH, "r") as fr:
                pts_full = fr['metadata']['plaintext'][:, BYTE_IDX].astype(np.uint8)
                keys_full = fr['metadata']['key'][:, BYTE_IDX].astype(np.uint8)
            pts = pts_full[Np:Np+Na]
            vals, cnts = np.unique(keys_full, return_counts=True)
            true_key = int(vals[np.argmax(cnts)])

            Xc = Xa - Xa.mean(axis=0, keepdims=True)
            y_std_eps = 1e-9

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
            m_values = list(range(10, Na+1, max(1, Na//50)))
            ranks_trials = np.zeros((len(m_values), N_TRIALS), dtype=np.float32)

            # 提前构造 HW 查表
            hw_lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.float32)

            for t in range(N_TRIALS):
                print(f"[CPA] Trial {t+1}/{N_TRIALS}")
                order = rng.permutation(Na)
                Xm_all = Xc[order].astype(np.float32)
                pts_ord = pts[order]
                for mi, m in enumerate(m_values):
                    if mi % 5 == 0:
                        print(f"  m={m} ({mi+1}/{len(m_values)})", flush=True)
                    Xm = Xm_all[:m, :]
                    Xm_std = Xm.std(axis=0) + 1e-9

                    scores = np.zeros(256, dtype=np.float32)
                    # 分批计算以节省内存
                    for start in range(0, 256, KEY_BATCH_SIZE):
                        end = min(start + KEY_BATCH_SIZE, 256)
                        k_range = np.arange(start, end, dtype=np.uint8)
                        # 计算所有 kguess 的 HW 模型值
                        pg = (pts_ord[:m, None] ^ k_range[None, :])
                        Lk_all = hw_lut[sbox[pg]]
                        Lk_all -= Lk_all.mean(axis=0, keepdims=True)
                        Lstd = Lk_all.std(axis=0, keepdims=True) + y_std_eps
                        Ln = Lk_all / Lstd
                        cov_all = (Ln.T @ Xm) / float(m)
                        corr_all = np.abs(cov_all / Xm_std[None, :])
                        scores[start:end] = corr_all.max(axis=1)

                    order_k = np.argsort(-scores)
                    ranks_trials[mi, t] = int(np.where(order_k == true_key)[0][0]) + 1

            ge = ranks_trials.mean(axis=1)
            sr = (ranks_trials == 1).mean(axis=1)
            ttd_list = []
            for t in range(N_TRIALS):
                hit = np.where(ranks_trials[:, t] == 1)[0]
                ttd_list.append(int(m_values[hit[0]]) if hit.size > 0 else None)

            # ======== 绘图与结果保存 ========
            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ge, marker='o')
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('#attack traces (m)')
            plt.ylabel('GE (avg rank)')
            plt.title('CPA GE curve (byte %d)' % BYTE_IDX)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "cpa_ge_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, sr, marker='o')
            plt.xlabel('#attack traces (m)')
            plt.ylabel('Success rate (rank=1)')
            plt.title('CPA SR curve (byte %d)' % BYTE_IDX)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "cpa_sr_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.plot(m_values, ranks_trials[:, 0], marker='o')
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('#attack traces (m)')
            plt.ylabel('Key rank (1=best)')
            plt.title('CPA key-rank curve (trial 0)')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, 'cpa_key_rank.png'), dpi=150)
            plt.close()

            valid_ttd = [v for v in ttd_list if v is not None]
            if len(valid_ttd) > 0:
                plt.figure(figsize=(5, 3))
                plt.hist(valid_ttd, bins=min(10, len(valid_ttd)))
                plt.xlabel('TTD (first m with rank=1)')
                plt.ylabel('count')
                plt.title('CPA TTD histogram (byte %d)' % BYTE_IDX)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, "cpa_ttd_hist.png"), dpi=150)
                plt.close()

            import csv
            with open(os.path.join(RESULT_DIR, "cpa_metrics.csv"), 'w', newline='') as fw:
                w = csv.writer(fw)
                w.writerow(["m", "GE", "SR"])
                for m, g, s in zip(m_values, ge, sr):
                    w.writerow([int(m), float(g), float(s)])

            with open(os.path.join(RESULT_DIR, "summary.txt"), "a", encoding="utf-8") as fw:
                fw.write(f"CPA metrics with raw data (byte {BYTE_IDX})\n")
                fw.write(f"GE@{m_values[-1]}={float(ge[-1]):.3f}, SR@{m_values[-1]}={float(sr[-1]):.3f}\n")
                if len(valid_ttd) > 0:
                    fw.write(f"TTD(mean/median)={np.mean(valid_ttd):.1f}/{np.median(valid_ttd):.1f}\n")
                fw.write(f"Recovered={'YES' if (sr[-1] > 0 or len(valid_ttd) > 0) else 'NO'}\n")

        except Exception as e:
            print("CPA 多指标评测失败：", e)


if __name__ == "__main__":
    main()
