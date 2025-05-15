import matplotlib.pyplot as plt
import numpy as np
import os

# Confusion matrix per ogni fold
conf_matrices = {
    'Fold 1': np.array([[283918, 2], [0, 181]]),
    'Fold 2': np.array([[284005, 0], [0, 183]])
}

# Etichette
labels = ['Actual Negative', 'Actual Positive']
pred_labels = ['Predicted Negative', 'Predicted Positive']

def plot_conf_matrix_matplotlib(matrix, fold_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap='Greens')
    plt.title(f'Confusion Matrix - {fold_name}', fontsize=14, fontweight='bold', pad=20)

    # Color bar
    fig.colorbar(cax)

    # Ticks e etichette
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(pred_labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotazioni numeriche
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val}', ha='center', va='center',
                color='black', fontsize=12, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()

    filename = os.path.join(f'{fold_name}_confusion_matrix.png')
    plt.savefig(filename, dpi=300)
    print(f"Salvata: {filename}")

# Plot per ogni fold
for fold, matrix in conf_matrices.items():
    plot_conf_matrix_matplotlib(matrix, fold)

