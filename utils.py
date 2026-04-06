'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import csv
import torch
import shutil
import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, recall_score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()



def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    #print('target', target, 'output', output)    
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print('Target: ', target, 'Pred: ', pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        #print(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        #print('F1: ', f1)
        return res, f1*100
    #print(res)
    return res



def save_checkpoint(state, is_best, opt, fold, fold_dir):
    torch.save(state, os.path.join(fold_dir, f'{opt.store_name}_checkpoint{fold}.pth'))
    if is_best:
        shutil.copyfile(os.path.join(fold_dir, f'{opt.store_name}_checkpoint{fold}.pth'),
                        os.path.join(fold_dir, f'{opt.store_name}_best{fold}.pth'))

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate

def save_csv_report(y_true, y_pred, fold_dir, fold, emotion_labels):
    report_dict = classification_report(y_true, y_pred, target_names=emotion_labels, digits=2, output_dict=True)
    report_csv_path = os.path.join(fold_dir, f"test_report_fold_{fold}.csv")
    with open(report_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
        for cls, metrics in report_dict.items():
            if cls not in emotion_labels:
                continue
            writer.writerow([
                cls,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"]
            ])

def get_emotion_labels(dataset_name):
    """
    Get emotion labels for the dataset
    """
    if dataset_name == 'RAVDESS':
        return [
            "neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprise"
        ]
    elif dataset_name == 'CREMAD':
        # CREMA-D has 6 emotions
        return [
            "neutral", 
            "happy",
            "sad",
            "fearful",
            "disgust",
            "angry"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
def save_confusion_matrix(y_true,y_pred,fold_dir, fold, emotion_labels):
    # ----- SAVE CONFUSION MATRIX IMAGE -----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – Fold {fold}")

    cm_path = os.path.join(fold_dir, f"confusion_matrix_fold_{fold}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.show()

def compute_war_uar(y_true, y_pred):
    """Return Weighted Average Recall (WAR) and Unweighted AR (UAR)."""
    war = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    uar = recall_score(y_true, y_pred, average='macro',    zero_division=0)
    return war, uar
def build_class_weights(dataset, num_classes, device):
    """Compute inverse-frequency class weights from a dataset."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    if hasattr(dataset, 'data'):
        for sample in dataset.data:
            counts[sample['label']] += 1
    weights = torch.zeros(num_classes, dtype=torch.float)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = 1.0 / counts[c].float()
    if weights.sum() > 0:
        weights = weights / weights.sum() * num_classes
    return weights.to(device)