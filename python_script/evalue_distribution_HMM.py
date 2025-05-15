import matplotlib.pyplot as plt
import numpy as np

""" Make a graphic for the E-value distribution obtained with HMM model in both positive and negative sets. 
    In logarithmic scale and with normalization to overcome the set's difference"""

file_pos1 = 'pos_testset/pos_1.class'
file_pos2 = 'pos_testset/pos_2.class'
file_neg1 = 'neg_testset/neg_1.class'
file_neg2 = 'neg_testset/neg_1.class'

def read_evalues(filepath, expected_label):
    evalues = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue  
            try:
                label = int(parts[1])
                evalue = float(parts[3])
                if label == expected_label:
                    if evalue == 0:
                        evalue = 1e-300 
                    evalues.append(evalue)
            except ValueError:
                continue
    return evalues


evalues_pos = read_evalues(file_pos1, 1) + read_evalues(file_pos2, 1)
evalues_neg = read_evalues(file_neg1, 0) + read_evalues(file_neg2, 0)

# Change the value in logarithmic scale 
log_pos = np.log10(evalues_pos)
log_neg = np.log10(evalues_neg)

# The density option provides the normalization of the data
plt.figure(figsize=(10, 6))
plt.hist(log_pos, bins=50, alpha=0.6, color='blue', label='Positive', density=True)
plt.hist(log_neg, bins=50, alpha=0.6, color='red', label='Negative', density=True)

plt.xlabel('Log10(E-value)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('E-value logarithmic distribution (HMMsearch)', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
