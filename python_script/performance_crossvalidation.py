import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

"""This script performs: 
- the confusion matrix
- MCC, Q2 (Accuracy), TPR (Recall), FPR, PPV (Precision)
- the MCC - E-value curves
- the ROC curves and AUC

=== INSTRUCTION TO RUN ON TERMINAL ===
python3 performace_crossvalidation.py set_1.class set_2.class total_set.class """

def get_cm(filename, threshold, pe=3, pr=1):
    """create the confusion matrix"""
    cm = [[0, 0], [0, 0]]  # TN, FN | FP, TP
    with open(filename) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < max(pe, pr) + 1:
                continue
            label = int(fields[pr])
            evalue = float(fields[pe])
            pred = 1 if evalue <= threshold else 0
            cm[pred][label] += 1
    return cm

def get_mcc(cm):
    """calculate the MCC"""
    tp = cm[1][1]
    tn = cm[0][0]
    fp = cm[1][0]
    fn = cm[0][1]
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
    return (tp * tn - fp * fn) / denom if denom else 0.0

def get_q2(cm):
    """calculate the Accuracy"""
    total = sum([sum(row) for row in cm])
    correct = cm[0][0] + cm[1][1]
    return correct / total if total > 0 else 0.0

def get_tpr(cm):
    """calculate the True Positive Rate (TPR)"""
    tp = cm[1][1]
    fn = cm[0][1]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def get_ppv(cm):
    """calculate the Predicted Positive Value (PPV)"""
    tp = cm[1][1]
    fp = cm[1][0]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def get_fpr(cm):
    """Calculate the False Positive Rate (FPR)"""
    fp = cm[1][0]
    tn = cm[0][0]
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def compute_curve(filename, start=-30, end=0, step=0.5):
    """Save all the evaluated threshold (E-Value) and all the MCC obtained"""
    thresholds = []
    mccs = []
    for i in range(int((end - start) / step) + 1):
        threshold = 10**(start + i * step)
        cm = get_cm(filename, threshold)
        mcc = get_mcc(cm)
        thresholds.append(threshold)
        mccs.append(mcc)
    return thresholds, mccs

def find_best_threshold(filename, start=-30, end=0, step=0.5):
    """function to find the best threshold correlated with the best MCC value"""
    best_mcc = -1.0
    best_thresh = None
    for i in range(int((end - start) / step) + 1):
        threshold = 10**(start + i * step)
        cm = get_cm(filename, threshold)
        mcc = get_mcc(cm)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = threshold
    return best_thresh

def compute_roc_curve(filename, start=-30, end=1, step=0.5):
    """Compute points for ROC curve"""
    fprs = []
    tprs = []
    for i in range(int((end - start) / step) + 1):
        threshold = 10**(start + i * step)
        cm = get_cm(filename, threshold)
        fprs.append(get_fpr(cm))
        tprs.append(get_tpr(cm))

    fprs = np.array(fprs)
    tprs = np.array(tprs)
    sorted_indices = np.argsort(fprs)
    fprs_sorted = fprs[sorted_indices]
    tprs_sorted = tprs[sorted_indices]

    auc_value = auc(fprs_sorted, tprs_sorted)
    return fprs_sorted.tolist(), tprs_sorted.tolist(), auc_value


def plot_roc_curve(fprs, tprs, auc_value):
    plt.figure(figsize=(6, 6))
    plt.plot(fprs, tprs, color='blue', label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random classification')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 performance_with_plot.py <set1.class> <set2.class>")
        sys.exit(1)

    set1 = sys.argv[1]
    set2 = sys.argv[2]
    total_set = sys.argv[3]

    out_txt = "hmm_results.txt"
    out_png = "mcc_vs_threshold.png"

    with open(out_txt, "w") as out:
        out.write("=== Fold 1: Train on Set1, Test on Set2 ===\n")
        t1 = find_best_threshold(set1)
        out.write(f"Best threshold from Set1: {t1:.2e}\n")
        cm1 = get_cm(set2, t1)
        out.write(f"Confusion Matrix on Set2: TN={cm1[0][0]} FN={cm1[0][1]} | FP={cm1[1][0]} TP={cm1[1][1]}\n")
        out.write(f"Accuracy (Q2): {get_q2(cm1):.4f}\n")
        out.write(f"MCC: {get_mcc(cm1):.4f}\n")
        out.write(f"Recall (TPR): {get_tpr(cm1):.4f}\n")
        out.write(f"Precision (PPV): {get_ppv(cm1):.4f}\n\n")

        out.write("=== Fold 2: Train on Set2, Test on Set1 ===\n")
        t2 = find_best_threshold(set2)
        out.write(f"Best threshold from Set2: {t2:.2e}\n")
        cm2 = get_cm(set1, t2)
        out.write(f"Confusion Matrix on Set1: TN={cm2[0][0]} FN={cm2[0][1]} | FP={cm2[1][0]} TP={cm2[1][1]}\n")
        out.write(f"Accuracy (Q2): {get_q2(cm2):.4f}\n")
        out.write(f"MCC: {get_mcc(cm2):.4f}\n")
        out.write(f"Recall (TPR): {get_tpr(cm2):.4f}\n")
        out.write(f"Precision (PPV): {get_ppv(cm2):.4f}\n\n")

        avg_mcc = (get_mcc(cm1) + get_mcc(cm2)) / 2
        avg_q2 = (get_q2(cm1) + get_q2(cm2)) / 2
        out.write("=== Summary ===\n")
        out.write(f"Average MCC: {avg_mcc:.4f}\n")
        out.write(f"Average Q2: {avg_q2:.4f}\n")

    # Compute the MCC-Evalue curves for both the sets
    thresholds, mccs = compute_curve(total_set)

    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mccs, linestyle='-', color='blue', alpha=0.8, label="Set1 (train)")

    plt.xscale("log")
    plt.xlabel("E-value Threshold (log scale)")
    plt.ylabel("MCC")
    plt.title("MCC vs E-value Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    # Compute ROC curve
    fprs, tprs, auc_val = compute_roc_curve(total_set)
    plot_roc_curve(fprs, tprs, auc_val)




