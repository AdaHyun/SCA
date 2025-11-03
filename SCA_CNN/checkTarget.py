# # check_h5_leakage.py
# import h5py
# import numpy as np

# AES_Sbox = np.array([
#     0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
#     0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
#     0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
#     0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
#     0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
#     0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
#     0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
#     0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
#     0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
#     0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
#     0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
#     0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
#     0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
#     0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
#     0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
#     0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
# ])

# def inspect_h5(filename, n_show=5):
#     with h5py.File(filename, 'r') as f:
#         print("Top-level groups:", list(f.keys()))
#         # try to show the profiling traces / labels
#         for p in ['Profiling_traces', 'Attack_traces']:
#             if p in f:
#                 grp = f[p]
#                 print(f"\n--- {p} keys:", list(grp.keys()))
#                 if 'traces' in grp:
#                     print(f"{p}/traces shape: {grp['traces'].shape}, dtype: {grp['traces'].dtype}")
#                 if 'labels' in grp:
#                     print(f"{p}/labels shape: {grp['labels'].shape}, dtype: {grp['labels'].dtype}")
#                 if 'metadata' in grp:
#                     print(f"{p}/metadata dtype: {grp['metadata'].dtype}")
#         # try to read small samples of metadata if present
#         if 'Profiling_traces' in f and 'metadata' in f['Profiling_traces']:
#             md = f['Profiling_traces/metadata']
#             print("\n--- Metadata sample (first entries) ---")
#             try:
#                 # If structured numpy array-like
#                 md_arr = np.array(md)
#                 print("metadata as ndarray; shape:", md_arr.shape, "dtype:", md_arr.dtype)
#                 # try to get plaintext and key fields
#                 if 'plaintext' in md_arr.dtype.names and 'key' in md_arr.dtype.names:
#                     print("plaintext[0]:", md_arr['plaintext'][0])
#                     print("key[0]:     ", md_arr['key'][0])
#             except Exception as e:
#                 print("Could not cast metadata to ndarray:", e)
#                 # try reading first element as object (list-of-dicts)
#                 try:
#                     first = md[0]
#                     print("metadata[0] (raw):", first)
#                     # try accessing plaintext/key if dict-like
#                     if isinstance(first, (dict,)) and 'plaintext' in first and 'key' in first:
#                         print("plaintext[0]:", first['plaintext'])
#                         print("key[0]:     ", first['key'])
#                 except Exception as e2:
#                     print("Could not read first metadata element:", e2)

#         # If labels exist, try to find which byte matches SBox(plaintext^key)
#         if 'Profiling_traces' in f and 'labels' in f['Profiling_traces'] and 'metadata' in f['Profiling_traces']:
#             labs = np.array(f['Profiling_traces/labels'])
#             md = f['Profiling_traces/metadata']
#             try:
#                 md_arr = np.array(md)
#                 # md_arr['plaintext'] -> shape (N, 16)
#                 plaintexts = md_arr['plaintext']
#                 keys = md_arr['key']
#                 n = min(len(labs), len(plaintexts), 200)  # check up to 200 samples
#                 print("\nChecking label match rates for each byte (using first %d samples):" % n)
#                 match_rates = []
#                 for b in range(plaintexts.shape[1]):
#                     svals = AES_Sbox[(plaintexts[:n, b] ^ keys[:n, b]).astype(np.int64)]
#                     matches = (svals == labs[:n])
#                     rate = matches.mean()
#                     match_rates.append(rate)
#                     print(f" byte {b}: match rate = {rate:.4f}")
#                 best = np.argmax(match_rates)
#                 print(f"\n=> Best matching byte: {best} (rate {match_rates[best]:.4f})")
#             except Exception as e:
#                 print("Failed matching labels with metadata ndarray:", e)
#                 # fallback: try list-like metadata
#                 try:
#                     n = min(len(labs), 200)
#                     plaintexts = np.array([md[i]['plaintext'] for i in range(n)])
#                     keys = np.array([md[i]['key'] for i in range(n)])
#                     print("\nChecking (fallback) label match rates for each byte:")
#                     for b in range(plaintexts.shape[1]):
#                         svals = AES_Sbox[(plaintexts[:, b] ^ keys[:, b]).astype(np.int64)]
#                         rate = (svals == labs[:n]).mean()
#                         print(f" byte {b}: match rate = {rate:.4f}")
#                 except Exception as e2:
#                     print("Fallback also failed:", e2)

#         # If metadata present but no labels, check if key is fixed (i.e. same across traces)
#         if 'Profiling_traces' in f and 'metadata' in f['Profiling_traces'] and 'labels' not in f['Profiling_traces']:
#             try:
#                 md_arr = np.array(f['Profiling_traces/metadata'])
#                 if 'key' in md_arr.dtype.names:
#                     keys = md_arr['key']  # shape (N,16)
#                     for b in range(keys.shape[1]):
#                         uniq = np.unique(keys[:, b])
#                         print(f"byte {b} unique keys count:", len(uniq), "example:", uniq[:5])
#                     print("如果每个字节的唯一 key 数量为 1，说明是 fixed-key 数据集。")
#             except Exception as e:
#                 print("无法检测 key 是否固定：", e)

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python checkTarget.py ASCAD_fixedkey.h5")
#         sys.exit(1)
#     fname = sys.argv[1]
#     inspect_h5(fname)

    



# # check_key_desync.py
# import h5py
# import numpy as np
# import sys

# h5path = sys.argv[1] if len(sys.argv)>1 else "/mnt/data/your.h5"

# with h5py.File(h5path, "r") as f:
#     print("Top groups:", list(f.keys()))
#     md = np.array(f['Profiling_traces/metadata'])
#     # byte index 可以改为任何你想检查的字节
#     byte_idx = 2
#     unique_keys = np.unique(md['key'][:, byte_idx])
#     unique_desync = np.unique(md['desync'])
#     print(f"Unique key values for byte {byte_idx}: {unique_keys} (count={len(unique_keys)})")
#     print("Example plaintext/key for first trace:", md['plaintext'][0], md['key'][0])
#     print("Unique desync values:", unique_desync)


# import h5py, numpy as np
# f = h5py.File("/root/autodl-tmp/SCA/data/processedData/ASCAD_fixedkey.h5", "r")
# keys = np.array(f["Profiling_traces/metadata"]["key"])
# print("Unique key values per byte:")
# for b in range(16):
#     uniq = np.unique(keys[:, b])
#     print(f"byte {b}: {len(uniq)} unique values")


# import h5py, numpy as np
# f = h5py.File("/root/autodl-tmp/SCA/data/processedData/ASCAD_fixedkey.h5", "r")
# Y = np.array(f['Profiling_traces/labels'])
# print("Y shape:", Y.shape)
# print("Unique values:", np.unique(Y)[:20])
# print("Unique count:", len(np.unique(Y)))

import h5py, numpy as np
f = h5py.File("/root/autodl-tmp/SCA/data/processedData/ASCAD_fixedkey.h5","r")
traces = np.array(f['Profiling_traces/traces'])
labels = np.array(f['Profiling_traces/labels'])
print("Trace shape:", traces.shape)
print("Mean/std over all traces:", np.mean(traces), np.std(traces))
print("Mean/std of first trace:", np.mean(traces[0]), np.std(traces[0]))
print("Label example:", labels[:10])
print("Std across traces at sample 300:", np.std(traces[:,300]))
