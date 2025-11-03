import h5py, numpy as np
import matplotlib.pyplot as plt

h5 = r"D:\Study\VScodeProject\SCA_CNN\ASCAD_fixedkey.h5"   # Windows: use double backslashes or quotes
# 示例: r"D:\Study\VScodeProject\SCA_CNN\ASCAD_fixedkey.h5"

with h5py.File(h5, "r") as f:
    traces = np.array(f['Profiling_traces/traces'], dtype=np.float32)  # read into memory
    labels = np.array(f['Profiling_traces/labels'])
print("traces shape:", traces.shape)
stds = np.std(traces, axis=0)
print("global mean/std:", traces.mean(), traces.std())
# plot (save)
plt.figure(figsize=(12,4))
plt.plot(stds, label='std across traces')
plt.xlabel("sample index")
plt.ylabel("std")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("quick_std_plot.png", dpi=150)
plt.show()

import h5py, numpy as np
f = h5py.File("D:\Study\VScodeProject\SCA_CNN\ASCAD_fixedkey.h5","r")
X = np.array(f['Profiling_traces/traces'])
Y = np.array(f['Profiling_traces/labels'])
print("traces shape:", X.shape)
print("labels shape:", Y.shape)
print("label unique (first 50):", np.unique(Y)[:50])
print("label unique count:", len(np.unique(Y)))
print("example plaintext/key for first trace:", np.array(f['Profiling_traces/metadata']['plaintext'][0],dtype=int),
      np.array(f['Profiling_traces/metadata']['key'][0],dtype=int))
f.close()


print("Saved quick_std_plot.png")
# optionally print top peaks
peaks_idx = np.argsort(stds)[-10:][::-1]
print("Top std indices (peak first):", peaks_idx)
print("Top std values:", stds[peaks_idx])

