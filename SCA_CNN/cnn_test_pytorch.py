#!/usr/bin/env python3
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
else:
    mpl.use('TkAgg')
import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ======== AES SBox (unchanged) ========
AES_Sbox = np.array([
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
])

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"Error: provided file path '{file_path}' does not exist!")
        sys.exit(-1)

# ======== PyTorch model (matching your CNN topology) ========
class CNNSimple(nn.Module):
    def __init__(self, input_dim=700, classes=256):
        super().__init__()
        # Convs (matches Keras structure: 5 conv blocks + avgpool)
        self.conv1 = nn.Conv1d(1, 64, 11, padding=5)
        self.pool1 = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 11, padding=5)
        self.pool2 = nn.AvgPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, 11, padding=5)
        self.pool3 = nn.AvgPool1d(2)
        self.conv4 = nn.Conv1d(256, 512, 11, padding=5)
        self.pool4 = nn.AvgPool1d(2)
        self.conv5 = nn.Conv1d(512, 512, 11, padding=5)
        self.pool5 = nn.AvgPool1d(2)
        self.relu = nn.ReLU()

        # fc layers (flatten after 5 pools => divide by 32)
        self.fc1 = nn.Linear((input_dim // 32) * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.pred = nn.Linear(4096, classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: [B, 1, L]
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.pred(x)
        x = torch.softmax(x, dim=1)   # ✅ 仅在推理时计算 softmax 概率
        return x


def load_sca_model(model_file, input_dim=700):
    check_file_exists(model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNSimple(input_dim=input_dim)
    # load_state_dict may raise if naming mismatch; caller should ensure correct file
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    model.to(device)
    print(f"[Info] Loaded PyTorch model: {model_file} on {device}")
    return model, device

# ======== Data loading: read traces and also load metadata into memory (NumPy) ========
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # IMPORTANT: read arrays into memory and return numpy arrays (so file can be closed)
    f = h5py.File(ascad_database_file, "r")
    X_profiling = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    Y_profiling = np.array(f['Profiling_traces/labels'])
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    Y_attack = np.array(f['Attack_traces/labels'])
    if load_metadata:
        # Read metadata into numpy arrays. metadata may be a structured array with fields.
        mp = np.array(f['Profiling_traces/metadata'])
        ma = np.array(f['Attack_traces/metadata'])
        f.close()
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (mp, ma)
    f.close()
    return (X_profiling, Y_profiling), (X_attack, Y_attack)

# ======== Rank functions (made robust to metadata formats) ========
def _get_plaintext_and_key_from_metadata(metadata, idx, target_byte):
    """
    Return tuple (plaintext_byte, key_byte) for trace index idx and target_byte.
    metadata may be:
      - a list/array of dict-like objects: metadata[idx]['plaintext'][target_byte]
      - a structured numpy array with fields 'plaintext' and 'key': metadata['plaintext'][idx][target_byte]
      - a numpy array with nested array fields
    """
    # Try structured numpy array (most likely when we loaded metadata into memory)
    try:
        # metadata is structured ndarray: metadata['plaintext'][idx] -> array of 16
        plaintext = metadata['plaintext'][idx][target_byte]
        key = metadata['key'][idx][target_byte]
        return int(plaintext), int(key)
    except Exception:
        pass

    # Try original style list-of-dicts
    try:
        plaintext = metadata[idx]['plaintext'][target_byte]
        key = metadata[idx]['key'][target_byte]
        return int(plaintext), int(key)
    except Exception:
        pass

    # Try dict-of-arrays (less common)
    try:
        plaintext = metadata['plaintext'][idx][target_byte]
        key = metadata['key'][idx][target_byte]
        return int(plaintext), int(key)
    except Exception:
        pass

    raise RuntimeError("Unsupported metadata format while fetching plaintext/key for index %d" % idx)

def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
    # predictions: array shape (n_traces, 256) giving class probs
    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256, dtype=np.float64)
        # key_bytes_proba[key_candidate] += np.log(float(predictions[p_idx, class_index]) + 1e-40)
    else:
        key_bytes_proba = last_key_bytes_proba

    for p_idx in range(max_trace_idx - min_trace_idx):
        global_idx = min_trace_idx + p_idx
        # get plaintext and (optionally) key (but here we only need plaintext)
        try:
            plaintext, _ = _get_plaintext_and_key_from_metadata(metadata, global_idx, target_byte)
        except Exception as e:
            print(f"[Error] cannot read metadata for trace {global_idx}: {e}")
            sys.exit(-1)
        for key_candidate in range(256):
            class_index = AES_Sbox[plaintext ^ key_candidate]
            proba = float(predictions[p_idx, class_index])
            # avoid log(0)
            key_bytes_proba[key_candidate] += np.log(proba if proba > 0.0 else 1e-36)

    sorted_proba = np.array([key_bytes_proba[a] for a in key_bytes_proba.argsort()[::-1]])
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return int(real_key_rank), key_bytes_proba

def full_ranks(predictions, dataset, metadata, min_trace_idx, max_trace_idx, rank_step, target_byte):
    # metadata can be structured ndarray or list-of-dicts
    # first get real_key (from metadata[0])
    try:
        # structured numpy array
        real_key = int(metadata['key'][0][target_byte])
    except Exception:
        try:
            real_key = int(metadata[0]['key'][target_byte])
        except Exception:
            # maybe dict-of-arrays
            real_key = int(metadata['key'][0][target_byte])

    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t_idx, out_i in zip(index, range(len(index))):
        rk, key_bytes_proba = rank(predictions[t_idx-rank_step:t_idx], metadata, real_key, t_idx-rank_step, t_idx, key_bytes_proba, target_byte)
        f_ranks[out_i] = [t_idx - min_trace_idx, rk]
    return f_ranks

# ======== Main test function (keeps behavior + output similar) ========
def check_model(model_file, ascad_database, num_traces=None, target_byte=2):
    # Load data and metadata (metadata loaded into memory)
    (Xp, Yp), (Xa, Ya), (Mp, Ma) = load_ascad(ascad_database, load_metadata=True)

    model, device = load_sca_model(model_file, input_dim=len(Xa[0]))

    # take subset
    Xa = Xa[:num_traces]
    # prepare for PyTorch: shape [N, 1, L]
    Xa_expand = np.expand_dims(Xa, axis=1)  # [N,1,L]
    X_tensor = torch.tensor(Xa_expand, dtype=torch.float32).to(device)

    print(f"[Info] Running inference on {len(Xa_expand)} traces...")
    preds = []
    batch_size = 200
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            out = model(X_tensor[i:i+batch_size])
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    print("[Info] Computing rank curve...")
    ranks = full_ranks(preds, None, Ma, 0, num_traces, 1, target_byte)

    # save results (CSV + PNG)
    model_base = os.path.splitext(os.path.basename(model_file))[0]
    results_dir = "/root/autodl-tmp/SCA/Attack/CNN_pytorch/results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{model_base}_rank.csv")
    png_path = os.path.join(results_dir, f"{model_base}_rank.png")
    # CSV: two columns (traces, rank)
    np.savetxt(csv_path, ranks, delimiter=",", header="traces,rank", fmt="%d", comments="")
    print(f"[Info] Saved rank CSV to {csv_path}")

    plt.figure(figsize=(8,6))
    plt.plot(ranks[:,0], ranks[:,1], label=model_base)
    plt.title(f"Performance of {model_base} against {os.path.basename(ascad_database)}")
    plt.xlabel("number of traces")
    plt.ylabel("rank")
    plt.grid(True)
    plt.legend()
    plt.savefig(png_path)
    print(f"[Info] Saved rank PNG to {png_path}")

# ======== CLI entry point (unchanged default params) ========
if __name__ == "__main__":
    model_file = "/root/autodl-tmp/SCA/Attack/CNN_pytorch/20000traces5000points_my_cnn_best_desync0_epochs100_batchsize128.pth"
    ascad_database = "/root/autodl-tmp/SCA/data/processedData/ASCAD_fixedkey.h5"
    num_traces = 10000
    target_byte = 2
    check_model(model_file, ascad_database, num_traces, target_byte)
    try:
        input("Press enter to exit ...")
    except SyntaxError:
        pass
