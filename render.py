import os
import pickle
import matplotlib.pyplot as plt

base_dir = "/home/user/MultimodalEmotionRecognition/results/pretrain_audio"
num_folds = 1

for fold in range(1, num_folds + 1):
    fold_dir = os.path.join(base_dir, f"fold_{fold}")
    train_loss_path = "train_loss_" + str(fold) +".pkl"
    val_loss_path = "val_loss_" + str(fold) +".pkl"
    train_acc_path = "train_acc_" + str(fold) +".pkl"
    val_acc_path = "val_acc_" + str(fold) +".pkl"
    train_uar_path = "train_uar_" + str(fold) +".pkl"
    val_uar_path = "val_uar_" + str(fold) +".pkl"
    # Load data
    train_loss = pickle.load(open(os.path.join(fold_dir, train_loss_path), "rb"))
    val_loss   = pickle.load(open(os.path.join(fold_dir, val_loss_path),   "rb"))
    train_acc  = pickle.load(open(os.path.join(fold_dir, train_acc_path),  "rb"))
    val_acc    = pickle.load(open(os.path.join(fold_dir, val_acc_path),    "rb"))
    train_uar = pickle.load(open(os.path.join(fold_dir, train_uar_path),    "rb"))
    val_uar = pickle.load(open(os.path.join(fold_dir, val_uar_path),    "rb"))
    # Convert to lists
    epochs = list(train_loss.keys())
    train_loss_vals = list(train_loss.values())
    val_loss_vals   = list(val_loss.values())
    train_acc_vals  = list(train_acc.values())
    val_acc_vals    = list(val_acc.values())
    train_uar_vals  = list(train_uar.values())
    val_uar_vals    = list(val_uar.values())
    # -----------------------
    # 1) LOSS PLOT
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_vals, label="Training Loss")
    plt.plot(epochs, val_loss_vals, label="Validation Loss")
    plt.title(f"Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(base_dir, f"fold_{fold}_loss_curve.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot → {loss_path}")

    # -----------------------
    # 2) ACCURACY PLOT
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_vals, label="Training Accuracy")
    plt.plot(epochs, val_acc_vals, label="Validation Accuracy")
    plt.title(f"Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(base_dir, f"fold_{fold}_acc_curve.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved acc plot → {acc_path}")
    # -----------------------
    # 3) UAR PLOT
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_uar_vals, label="Training UAR")
    plt.plot(epochs, val_uar_vals, label="Validation UAR")
    plt.title(f"UAR Curve")
    plt.xlabel("Epoch")
    plt.ylabel("UAR")
    plt.legend()
    plt.grid(True)

    acc_path = os.path.join(base_dir, f"fold_{fold}_uar_curve.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved uar plot → {acc_path}")
