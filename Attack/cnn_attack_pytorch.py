#!/usr/bin/env python3
# cnn_attack_pytorch_compatible.py
# PyTorch替换版（保持原脚本结构与输出风格）

import os
import os.path
import sys
import h5py
import numpy as np
import math
import time

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For pretty printing the model summary
from collections import OrderedDict

# -------------------------
# Utility functions (unchanged interface)
# -------------------------
def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

# -------------------------
# ASCAD loader (exactly like before)
# -------------------------
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

# -------------------------
# PyTorch Dataset wrapper (keeps interface local)
# -------------------------
class ASCADDataset(Dataset):
    def __init__(self, X_numpy, Y_numpy, for_cnn=True):
        """
        X_numpy: numpy array shape [N, trace_len]
        Y_numpy: numpy array shape [N] (labels 0..255)
        for_cnn: if True reshape to [N, 1, trace_len], else keep [N, trace_len]
        """
        # [改动] 统一将原始 int8 波形转换为 float32 并做逐样本标准化，提高可训练性
        self.X = X_numpy.astype(np.float32)
        self.Y = Y_numpy.astype(np.int64)
        self.for_cnn = for_cnn
        if self.for_cnn:
            # reshape for CNN input: [N, 1, L]
            self.X = np.expand_dims(self.X, axis=1)
        # [改动] 逐样本标准化 (x - mean) / std，避免不同幅度尺度差异影响训练
        eps = 1e-6
        if self.X.ndim == 3:  # CNN: [N, 1, L]
            mean = self.X.mean(axis=-1, keepdims=True)
            std = self.X.std(axis=-1, keepdims=True) + eps
            self.X = (self.X - mean) / std
        elif self.X.ndim == 2:  # MLP: [N, L]
            mean = self.X.mean(axis=-1, keepdims=True)
            std = self.X.std(axis=-1, keepdims=True) + eps
            self.X = (self.X - mean) / std
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return torch.from_numpy(x), torch.from_numpy(np.array(y, dtype=np.int64))

# -------------------------
# Models - PyTorch implementations of original Keras topologies
# Keep function names (mlp_best, cnn_best, cnn_best2) similar to original.
# -------------------------
class MLPBest(nn.Module):
    def __init__(self, input_dim=1400, node=200, layer_nb=6, classes=256):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, node))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(layer_nb-2):
            layers.append(nn.Linear(node, node))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(node, classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [B, input_dim]
        return self.net(x)

def mlp_best(node=200, layer_nb=6, input_dim=1400):
    model = MLPBest(input_dim=input_dim, node=node, layer_nb=layer_nb, classes=256)
    return model

class CNNBestPT(nn.Module):
    def __init__(self, input_len=700, classes=256):
        super().__init__()
        # Mirror the Keras cnn_best (VGG-like 1D)
        # [改动] 为每个卷积块加入 BatchNorm1d，稳定梯度，加速收敛
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, padding=5)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=11, padding=5)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=11, padding=5)
        self.bn3   = nn.BatchNorm1d(256)
        self.pool3 = nn.AvgPool1d(2, stride=2)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=11, padding=5)
        self.bn4   = nn.BatchNorm1d(512)
        self.pool4 = nn.AvgPool1d(2, stride=2)
        
        self.conv5 = nn.Conv1d(512, 512, kernel_size=11, padding=5)
        self.bn5   = nn.BatchNorm1d(512)
        self.pool5 = nn.AvgPool1d(2, stride=2)

        # compute flattened feature length: simulate a forward pass length calc
        L = input_len
        for _ in range(5):
            # conv with padding='same' keeps length, pool halves (floor)
            L = math.floor((L + 1e-9) / 2)
        # L should be roughly 21 for input_len=700
        flat_len = 512 * L
        self.fc1 = nn.Linear(flat_len, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        # [改动] FC 段加入 Dropout，降低过拟合
        self.dropout_fc = nn.Dropout(p=0.5)
        self.pred = nn.Linear(4096, classes)

    def forward(self, x):
        # [改动] Conv -> BN -> ReLU -> Pool
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x))); x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x))); x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)  # [改动]
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)  # [改动]
        x = self.pred(x)
        return x

def cnn_best(classes=256, input_dim=700):
    return CNNBestPT(input_len=input_dim, classes=classes)

def cnn_best2(classes=256, input_dim=1400):
    # mimic cnn_best2 (strided first conv). Keep same architecture idea.
    model = CNNBestPT(input_len=input_dim, classes=classes)
    return model

# -------------------------
# helper: pretty print a model summary (minimal, self-contained)
# -------------------------
def print_model_summary_pytorch(model, input_shape, device):
    """
    model: nn.Module
    input_shape: tuple, e.g. (1, 700, 1) or (1, 1, 700) --> we will adapt to (B, C, L)
    This function runs a single forward with hooks to collect shapes and params.
    """
    # Decide a canonical tensor shape for model: for our models expect (B, 1, L)
    # If user passed (None, L, 1) we convert to (1, 1, L)
    # If input_shape is (None, L) -> MLP -> (1, L)
    if len(input_shape) == 3:
        # Might be (None, L, 1) in Keras style -> convert to (1, 1, L)
        if input_shape[2] == 1:
            sample_shape = (1, 1, input_shape[1])
        else:
            sample_shape = (1, input_shape[1], input_shape[2])
    elif len(input_shape) == 2:
        # (None, L) -> MLP input
        sample_shape = (1, input_shape[1])
    else:
        # fallback
        sample_shape = (1, 1, input_shape[-1])

    device = device if device is not None else torch.device('cpu')
    model = model.to(device)
    summary = []
    hooks = []

    def register_hook(module):
        def hook(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            # attempt to get param count
            params = 0
            for p in module.parameters(recurse=False):
                params += p.numel()
            out_shape = None
            if isinstance(output, (list, tuple)):
                out_shape = [tuple(o.size()) for o in output]
            else:
                try:
                    out_shape = tuple(output.size())
                except:
                    out_shape = str(type(output))
            summary.append((m_key, class_name, out_shape, params))
        if (not isinstance(module, nn.Sequential)) and (not isinstance(module, nn.ModuleList)) and (module != model):
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)

    # forward a dummy input
    try:
        x = torch.zeros(sample_shape).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(x)
    except Exception as e:
        # if forward fails, still try to provide rough param summary
        pass
    finally:
        for h in hooks:
            h.remove()

    # build human readable table
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel summary (PyTorch):")
    print("{:30} {:30} {:20} {:15}".format("Layer (type)", "Output Shape", "Param #", ""))
    print("="*100)
    # We will print the layers we captured
    for (name, cls, out_shape, params) in summary:
        print("{:30} {:30} {:20}".format(f"{name} ({cls})", str(out_shape), f"{params:,}"))
    print("-"*100)
    print("Total params: {:,}".format(total_params))
    print("Trainable params: {:,}".format(trainable_params))
    print("Non-trainable params: {:,}".format(total_params - trainable_params))
    print("")

# -------------------------
# Training function (keeps signature of original train_model)
# -------------------------
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100,
                multilabel=0, validation_split=0, early_stopping=0):
    """
    PyTorch training function but kept with original signature/behavior.
    - X_profiling: numpy array [N, trace_len]
    - Y_profiling: numpy array [N]
    - model: PyTorch nn.Module
    - save_file_name: path for saving .pth
    """
    # [改动] 确保保存目录存在（原逻辑为不存在即退出，这里改为自动创建）
    save_dir = os.path.dirname(save_file_name)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}{' ('+torch.cuda.get_device_name(0)+')' if torch.cuda.is_available() else ''}")

    # Determine if model expects CNN input (3D) or MLP input (2D) by checking first layer
    is_cnn = False
    try:
        # if first module is Conv1d => CNN
        first_children = list(model.children())
        if len(first_children) > 0 and isinstance(first_children[0], nn.Conv1d):
            is_cnn = True
    except Exception:
        is_cnn = True  # safe default

    # Dataset / DataLoader
    train_dataset = ASCADDataset(X_profiling, Y_profiling, for_cnn=is_cnn)
    # If validation_split > 0, create split (train/val)
    if validation_split and validation_split > 0:
        total = len(train_dataset)
        val_count = int(total * validation_split)
        train_count = total - val_count
        train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_count, val_count])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = None

    # prepare model & optimizer & loss
    model = model.to(device)
    # [改动] 使用 AdamW 优化器并提高学习率；原 LR=1e-5 过小导致几乎不学习
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # [改动] 使用 label smoothing（若版本不支持则回退）
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    # [改动] 余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    # Print model summary in a friendly way similar to Keras
    # Build an input_shape like original (None, L, 1) or (None, L)
    sample_shape = X_profiling.shape[1]
    if is_cnn:
        keras_style_input_shape = (None, sample_shape, 1)
    else:
        keras_style_input_shape = (None, sample_shape)
    print_model_summary_pytorch(model, keras_style_input_shape, device)

    # Training loop
    best_val_loss = None
    print(f"[Train] Starting training for {epochs} epochs, batch_size={batch_size}")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        t0 = time.time()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True).float()
            batch_y = batch_y.to(device, non_blocking=True).long().squeeze()

            # If MLP: batch_X shape [B, L] required; our dataset leaves it as [B, L]
            if not is_cnn and batch_X.dim() == 3:
                batch_X = batch_X.view(batch_X.size(0), -1)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_seen += batch_X.size(0)

        # compute epoch metrics
        avg_loss = epoch_loss / epoch_seen if epoch_seen else 0.0
        acc = epoch_correct / epoch_seen if epoch_seen else 0.0
        t1 = time.time()

        # Validation if any
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_seen = 0
            with torch.no_grad():
                for vX, vy in val_loader:
                    vX = vX.to(device, non_blocking=True).float()
                    vy = vy.to(device, non_blocking=True).long().squeeze()
                    if not is_cnn and vX.dim() == 3:
                        vX = vX.view(vX.size(0), -1)
                    vout = model(vX)
                    vloss = criterion(vout, vy)
                    val_loss_sum += vloss.item() * vX.size(0)
                    val_seen += vX.size(0)
            val_loss = val_loss_sum / val_seen if val_seen else None

        # Save model each epoch (mimic ModelCheckpoint default)
        try:
            torch.save(model.state_dict(), save_file_name)
        except Exception as e:
            print(f"[Warning] Could not save model to {save_file_name}: {e}")

        # Optionally implement early stopping based on val_loss if early_stopping != 0
        if early_stopping != 0 and val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                # save best
                try:
                    torch.save(model.state_dict(), save_file_name + ".best.pth")
                except:
                    pass

        # print epoch summary (match Keras-style console)
        print(f"Epoch {epoch+1}/{epochs}")
        # We attempt to mimic lines like "250/250 .... accuracy: 0.0036 - loss: 5.548"
        # We'll print a compact line:
        print(f"{epoch_seen}/{epoch_seen} - {int(t1-t0)}s {int((t1-t0)/max(1,epoch_seen//batch_size))}s/step - accuracy: {acc:.4f} - loss: {avg_loss:.4f}", end="")
        if val_loss is not None:
            print(f" - val_loss: {val_loss:.4f}")
        else:
            print("")
        # [改动] 每个 epoch 后步进一次学习率调度器
        try:
            scheduler.step()
        except Exception:
            pass
    print(f"[Train] Done. Model saved at: {save_file_name}")
    return None

# -------------------------
# Parameter file reader (same interface)
# -------------------------
def read_parameters_from_file(param_filename):
    #read parameters for the train_model and load_ascad functions
    #TODO: sanity checks on parameters
    param_file = open(param_filename,"r")
    #TODO: replace eval() by ast.literal_eval for safety? keep eval for parity with original
    my_parameters= eval(param_file.read())

    ascad_database = my_parameters["ascad_database"]
    training_model = my_parameters["training_model"]
    network_type = my_parameters["network_type"]
    epochs = my_parameters["epochs"]
    batch_size = my_parameters["batch_size"]
    train_len = 0
    if ("train_len" in my_parameters):
        train_len = my_parameters["train_len"]
    validation_split = 0
    if ("validation_split" in my_parameters):
        validation_split = my_parameters["validation_split"]
    multilabel = 0
    if ("multilabel" in my_parameters):
        multilabel = my_parameters["multilabel"]
    early_stopping = 0
    if ("early_stopping" in my_parameters):
        early_stopping = my_parameters["early_stopping"]
    return ascad_database, training_model, network_type, epochs, batch_size, train_len, validation_split, multilabel, early_stopping

# -------------------------
# Main: preserved control flow and defaults similar to your original script
# -------------------------
if __name__ == "__main__":
    if len(sys.argv)!=2:
        #default parameters values
        # [改动] 修改默认数据集路径为用户提供的本地路径
        ascad_database = r"/root/autodl-tmp/SCA/data/processedData/ASCAD_fixedkey.h5"
        #MLP training
        # network_type = "mlp"
        # [改动] 修改默认模型保存路径到当前工程下，避免 Linux 路径在 Windows 上不存在
        training_model = os.path.join(os.path.dirname(__file__), "CNN", "pytorch_my_cnn_best_desync0_epochs75_batchsize200.pth")

        #CNN training
        network_type = "cnn"
        validation_split = 0
        multilabel = 0
        train_len = 0
        epochs = 75
        batch_size = 200
        bugfix = 0
        early_stopping = 0
    else:
        #get parameters from user input
        ascad_database, training_model, network_type, epochs, batch_size, train_len, validation_split, multilabel, early_stopping = read_parameters_from_file(sys.argv[1])

    # load traces
    (X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database)

    # pick the network
    if(network_type=="mlp"):
        best_model = mlp_best(input_dim=len(X_profiling[0]))
    elif(network_type=="cnn"):
        best_model = cnn_best(input_dim=len(X_profiling[0]))
    elif(network_type=="cnn2"):
        best_model = cnn_best2(input_dim=len(X_profiling[0]))
    else:
        print("Error: no topology found for network '%s' ..." % network_type)
        sys.exit(-1)

    # Print a concise representation similar to Keras' summary (we will also print detailed summary inside train)
    print("[Info] Model created:", best_model.__class__.__name__)

    # Training
    if (train_len == 0):
        train_model(X_profiling, Y_profiling, best_model, training_model, epochs, batch_size, multilabel, validation_split, early_stopping)
    else:
        train_model(X_profiling[:train_len], Y_profiling[:train_len], best_model, training_model, epochs, batch_size, multilabel, validation_split, early_stopping)
