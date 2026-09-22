import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(train_losses, test_accs, save_path='.', file_name="training_curves.png", show=True):
    """
    draw traning curve：Loss + Accuracy
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(train_losses, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    # test_accs sample by 10 epoches
    epochs = range(0, len(train_losses), 10)
    if len(epochs) > len(test_accs):
        epochs = epochs[:len(test_accs)]
    ax2.plot(epochs, test_accs, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")
    

def plot_training_curves_f1(train_losses, f1s, save_path='.', file_name="training_curves_f1.png", show=True):
    """
    draw training curve：Loss + Accuracy
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(train_losses, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    # test_accs sample per 10 epoch
    epochs = range(0, len(train_losses), 10)
    if len(epochs) > len(f1s):
        epochs = epochs[:len(f1s)]
    ax2.plot(epochs, f1s, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy(F1 hit rate)')
    ax2.set_title('Test Accuracy(F1 hit rate)')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5, label='Random guess')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")    
    
    
    


def plot_grokking_analysis(test_accs, gradient_norms, train_losses=None, 
                          save_path='.', file_name='grokking_analysis.png', show=True):
    """
    Grokking Analysis：accuracy + grad nom
    
    Args:
        test_accs: sampled per 10 epoches
        gradient_norms: grad norm list
        train_losses: optional , losses per epoch list
        save_path: save path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # left：accuracy curve（label grokking position）
    epochs_acc = range(0, len(test_accs) * 10, 10)
    if len(epochs_acc) > len(test_accs):
        epochs_acc = epochs_acc[:len(test_accs)]
    
    ax1.plot(epochs_acc, test_accs, 'b-', linewidth=2, label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Grokking: Test Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
    
    # find  position accuracy jumps（从 < 0.9 到 > 0.9）
    grokking_epoch = None
    for i, acc in enumerate(test_accs):
        if acc > 0.9:
            grokking_epoch = i * 10
            break
    if grokking_epoch is not None:
        ax1.axvline(x=grokking_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Grokking at epoch {grokking_epoch}')
    ax1.legend()
    
    # left：gradnorm curve
    ax2.plot(gradient_norms, 'g-', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm Dynamics')
    ax2.grid(True, alpha=0.3)
    
    # should label  early peaking position
    if gradient_norms:
        grad = np.asarray(gradient_norms, dtype=float)
        dgrad = np.abs(np.diff(grad))

        # Robust threshold: median + k * MAD
        med = np.median(dgrad)
        mad = np.median(np.abs(dgrad - med)) + 1e-9
        threshold = med + 10 * mad

        # First index where |ΔGradNorm| exceeds the threshold
        candidates = np.where(dgrad > threshold)[0]
        if len(candidates) > 1:
            first_spike = int(candidates[1])
            ax2.axvline(
                x=first_spike, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7,
                label=f'First ΔGradNorm(Gradient spike) outlier at epoch {first_spike}',
            )

            
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_gradient_norm(gradient_norms, save_path='.',file_path='gradient_norm.png', show=True):
    """
    draw gradnorm: figure 4
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    plt.plot(gradient_norms, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm During Training')
    plt.grid(True, alpha=0.3)
    
    # should label  early peaking position
    if gradient_norms:
        grad = np.asarray(gradient_norms, dtype=float)
        dgrad = np.abs(np.diff(grad))

        # Robust threshold: median + k * MAD
        med = np.median(dgrad)
        mad = np.median(np.abs(dgrad - med)) + 1e-9
        threshold = med + 10 * mad

        # First index where |ΔGradNorm| exceeds the threshold
        candidates = np.where(dgrad > threshold)[0]
        if len(candidates) > 1:
            first_spike = int(candidates[1])
            ax2.axvline(
                x=first_spike, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7,
                label=f'First ΔGradNorm(Gradient spike) outlier at epoch {first_spike}',
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")
    
def plot_grokking_analysis_f1(test_f1s, gradient_norms, train_losses=None, 
                          save_path='.', file_name='grokking_analysis_f1.png', show=True):
    """
    draw Grokking analysis：F1 + gradnorm
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # left：F1
    epochs_f1 = range(0, len(test_f1s) * 10, 10)
    if len(epochs_f1) > len(test_f1s):
        epochs_f1 = epochs_f1[:len(test_f1s)]
    
    ax1.plot(epochs_f1, test_f1s, 'b-', linewidth=2, label='F1 Score')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Grokking: F1 Score')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    grokking_epoch = None
    for i, f1 in enumerate(test_f1s):
        if f1 >= 0.9:
            grokking_epoch = i * 10
            break
    if grokking_epoch is not None:
        ax1.axvline(x=grokking_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Grokking at epoch {grokking_epoch}')
    ax1.legend()
    
    # right：grad norm curve
    ax2.plot(gradient_norms, 'g-', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm Dynamics')
    ax2.grid(True, alpha=0.3)
    
    if gradient_norms:
        grad = np.asarray(gradient_norms, dtype=float)
        dgrad = np.abs(np.diff(grad))

        # Robust threshold: median + k * MAD
        med = np.median(dgrad)
        mad = np.median(np.abs(dgrad - med)) + 1e-9
        threshold = med + 10 * mad

        # First index where |ΔGradNorm| exceeds the threshold
        candidates = np.where(dgrad > threshold)[0]
        if len(candidates) > 1:
            first_spike = int(candidates[1])
            ax2.axvline(
                x=first_spike, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7,
                label=f'First ΔGradNorm(Gradient spike) outlier at epoch {first_spike}',
            )

    ax2.legend()
    
    plt.tight_layout()
    full_path = os.path.join(save_path, file_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {full_path}")