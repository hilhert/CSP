import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import matplotlib.pyplot as plt

from csp import (
    CSP, VanillaRNN, ComplexRNN, TransformerModel, MambaModel,
    generate_parity_data, create_dataloaders,
)
from utils import (
    train_model, evaluate_model, evaluate_model_f1, load_checkpoint,
    setup_logging, plot_training_curves, plot_grokking_analysis,set_seed
)


# ============================================================
# Config
# ============================================================
task = "parity"
hidden_dim = 32
train_len = 16
train_samples = 10000
num_layers = 3
batch_size = 64
output_dim = 2
max_len = 64
vocab_size= 2
embed_dim = 16
eval_lens = list(range(8, 65, 2))
eval_ = False
train_all = True
test_samples = 20000


base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, f"{task}/results")
os.makedirs(result_dir, exist_ok=True)


# ============================================================
# Models and their parameter dicts
# ============================================================
models = [
    CSP,
    CSP,
    VanillaRNN,
    ComplexRNN,
    TransformerModel,
    MambaModel,
]

m_params = [
    {
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "rope_input": False,
        "mode": "vanilla",
    },  # CSP vanilla
    {
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,        
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "rope_input": False,
        "mode": "fast",
    },  # CSP fast
    {
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,       
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
    },  # VanillaRNN
    {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
    },  # ComplexRNN
    {
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,       
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "n_heads": 4,
        "n_layers": num_layers,
        "max_len": max_len,
    },  # TransformerModel
    {
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,       
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "n_layers": num_layers,
    },  # MambaModel
]

model_labels = [
    "CSP-Vanilla",
    "CSP-Fast",
    "VanillaRNN",
    "ComplexRNN",
    "Transformer",
    "Mamba",
]


# ============================================================
# Main
# ============================================================
def main():
    all_results = {}
    print("=" * 60)
    print(f"Task: {task}")
    print("=" * 60)
    # Baselines first, CSP last
    zipped = list(zip(model_labels, models, m_params))
    
    SEEDS = [42, 43, 44, 45, 46]
 
    current_model = ""
    for seed,idx in enumerate(SEEDS):
        all_results_one_seed = {}
        for label, Model, Param in reversed(zipped):
            print("=" * 60)
            print(f"Model: {label}")
            print("=" * 60)

            model_path = os.path.join(base_dir, f"{task}/{label}")
            fig_path = os.path.join(model_path, "figure")
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(fig_path, exist_ok=True)

            cp_name = f"{task}_{label}_{hidden_dim}_{num_layers}_seed_{idx}.pt"
            cp_path = os.path.join(model_path, cp_name)
            
            if not train_all:
                if not os.path.exists(cp_path) and not eval_:
                    current_model = label

                if current_model != label:
                    continue

            
            log_file = setup_logging()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            set_seed(seed)
            g = torch.Generator().manual_seed(seed)
            model = Model(**Param).to(device)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}")

            # ---------- Train or load ----------
            if not eval_:
                X, y = generate_parity_data(train_samples, train_len)
                train_loader, test_loader = create_dataloaders(
                    X, y,generator=g, batch_size=batch_size)

                losses, _, accs, grad_norms = train_model(
                    model, train_loader, test_loader,
                    weight_decay=1e-7, epochs=300, lr=1e-4,
                    device=device, cp_path=cp_path, eval_=eval_,
                )

                plot_training_curves(losses, accs, save_path=fig_path)
                plot_grokking_analysis(accs, grad_norms, losses,
                                       save_path=fig_path)

                print(f"Train Acc with length {train_len}: "
                      f"{evaluate_model(model, test_loader, device):.4f}")
                print(f"Train F1 with length {train_len}: "
                      f"{evaluate_model_f1(model, test_loader, device):.4f}")
                print(f"Log: {log_file}")
            else:
                load_checkpoint(cp_path, model, optimizer=None, device=device)

            # ---------- Length generalization sweep ----------
            accus, f1s = [], []
            for len_ in eval_lens:
                X, y = generate_parity_data(test_samples, len_)
                _, test_loader = create_dataloaders(
                    X, y, generator=g,batch_size=batch_size)

                accu_ = evaluate_model(model, test_loader, device)
                f1_ = evaluate_model_f1(model, test_loader, device)

                accus.append(accu_)
                f1s.append(f1_)

                print(f"Eval length {len_:>3d} | "
                      f"Acc: {accu_:.4f} | F1: {f1_:.4f}")

            all_results_one_seed[label] = {
                "params": n_params,
                "eval_lens": eval_lens,
                "acc": accus,
                "f1": f1s,
            }
            # ---------- Save per-model results ----------
            with open(os.path.join(model_path, f"results_seed_{idx}.json"), "w") as f:
                json.dump(all_results_one_seed[label], f, indent=2)
            # ---------- Per-model length curve ----------
            plt.figure(figsize=(6, 4))
            plt.plot(eval_lens, accus, marker="o", label="Acc")
            plt.plot(eval_lens, f1s, marker="s", label="F1")
            plt.xlabel("Evaluation sequence length")
            plt.ylabel("Score")
            plt.title(f"{label} length generalization")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_path, f"length_generalization_seed_{idx}.png"),
                        dpi=200)
            plt.close()
            
        all_results[f"seed_{idx}"] = all_results_one_seed
            
        # ---------- Comparison plot: accuracy ----------
        plt.figure(figsize=(8, 5))
        for label, res in all_results[f"seed_{idx}"].items():
            plt.plot(res["eval_lens"], res["acc"], marker="o", label=label)
        plt.xlabel("Evaluation sequence length")
        plt.ylabel("Accuracy")
        plt.title(f"{task}: length generalization (accuracy)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{task}_length_acc_seed_{idx}.png"), dpi=200)
        plt.close()

        # ---------- Comparison plot: F1 ----------
        plt.figure(figsize=(8, 5))
        for label, res in all_results[f"seed_{idx}"].items():
            plt.plot(res["eval_lens"], res["f1"], marker="s", label=label)
        plt.xlabel("Evaluation sequence length")
        plt.ylabel("F1")
        plt.title(f"{task}: length generalization (F1)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{task}_length_f1_seed_{idx}.png"), dpi=200)
        plt.close()
    # ---------- Save all results ----------
    with open(os.path.join(result_dir, f"{task}_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("All done. Results saved to", result_dir)


if __name__ == "__main__":
    main()