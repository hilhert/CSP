import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from safetensors.torch import save_model, save_file
import os
from sklearn.metrics import f1_score
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def save_checkpoint(model, optimizer, epoch, loss, f1, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'f1': f1,
    }
    # save .pt train state dict all 
    torch.save(checkpoint, filepath)
    
    #  pure_name.safetensors
    base_path, _ = os.path.splitext(filepath)
    
    has_rnn = any(k.startswith("rnn.") for k in model.state_dict().keys())
    if not has_rnn:
        safetensors_path = base_path + ".safetensors"
        for module in model.modules():
            if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                module.flatten_parameters = lambda: None
        save_model(model, safetensors_path)
    
def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    [recovering]  load model and optimizer check point form .pt file 
    
    param:
        filepath: .pt checkpoint  (for example "checkpoints/best_model.pt")
        model:  PyTorch modal already have been initialized
        optimizer: the optimizer together with the model state dict (optional)
        
    return:
        model, optimizer, start_epoch, f1
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find checkpoint file: {filepath}")
        
    print(f"[Checkpoint] recovering training state: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    # 1. recover model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. revocer optimizer state (if  optimizer have been passed)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1  # continue_with the next epoch
    loss = checkpoint.get('loss', 0.0)
    f1 = checkpoint.get('f1', 0.0)
    
    print(f"[Checkpoint] successfully recovered to  Epoch {start_epoch} | last Loss: {loss:.4f} | F1: {f1:.4f}")
    return model, optimizer, start_epoch, f1


def load_model(filepath, model, device='cpu'):
    """
    【reasoning/evaluating】load model weights
    Try to load .safetensors firstly, if it does not exists, load .pt file
    
    param:
        filepath: path contaings the model (ie "best_model.safetensors" or "best_model.pt")
        model:  PyTorch model already been initialized
        device:  ('cpu', 'cuda' )
        
    return:
        model (the eval state)
    """
    base_path, _ = os.path.splitext(filepath)
    safetensors_path = base_path + ".safetensors"
    pt_path = base_path + ".pt"
    
    model = model.to(device)
    
    # find .safetensors fistrly
    if os.path.exists(safetensors_path):
        print(f"[Model] load model using Safetensors format: {safetensors_path}")
        st_load_model(model, safetensors_path)
    # find .pt secondary
    elif os.path.exists(pt_path):
        print(f"[Model] could not find Safetensors model file, roll back to PyTorch .pt loading: {pt_path}")
        checkpoint = torch.load(pt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"could not find checkpoint with both formats (.safetensors or .pt): {base_path}")
        
    model.eval()  # switch to eval mode
    print(f"[Model] widht already been loaded, ready to eval！")
    return model    


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    ce = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def evaluate_model(model, test_loader, device='cpu', ignore_index=-100):
    """
    Token-level accuracy over all non-ignored positions.

    model output: [B, T, C]
    target:       [B, T]  with ignore_index on padding
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)                # [B, T, C]
            preds = logits.argmax(dim=-1)          # [B, T]

            mask = (y_batch != ignore_index)
            correct += (preds[mask] == y_batch[mask]).sum().item()
            total += mask.sum().item()

    return correct / max(total, 1)


def evaluate_model_f1(model, test_loader, device='cpu', ignore_index=-100):
    """
    Token-level F1 over all non-ignored positions.
    For binary labels, use average='binary'.
    For multi-class labels, use average='macro' or 'micro'.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=-1)          # [B, T]

            mask = (y_batch != ignore_index)
            all_preds.extend(preds[mask].cpu().numpy())
            all_targets.extend(y_batch[mask].cpu().numpy())
    num_classes = len(set(all_targets))
    avg = 'binary' if num_classes == 2 else 'macro'
    return f1_score(all_targets, all_preds, average=avg)


def train_model(model, train_loader, test_loader, epochs=300,weight_decay=1e-8,lr=0.001, device='cpu', logger=None, loss_fn=None, ignored_idx=-100, cp_path=None, eval_= False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(ignore_index = ignored_idx,reduction='none')
    
    train_losses, f1s,test_accs, gradient_norms = [],[],[], []
    best_f1 = 0.0
    best_accu = 0.0
    log = logger.info if logger else print
    
    for epoch in tqdm(range(epochs), desc="Training"):
       
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = torch.mean(loss_fn(pred.permute(0,2,1), y_batch))

            optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        gradient_norms.append(total_norm)
        
       
        if epoch % 10 == 0:
            test_acc = evaluate_model(model, test_loader, device)
            f1       = evaluate_model_f1(model, test_loader, device)
            test_accs.append(test_acc)
            f1s.append(f1)
            #scheduler.step(test_acc)
            if  f1 >= best_f1 and test_acc >= best_accu:
                save_checkpoint(model, optimizer, epoch, loss, f1, cp_path)
                best_f1 = f1
                best_accu = test_acc
            log(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {test_acc:.4f} | F1: {f1:.4f} | GradNorm: {total_norm:.4f} | Best_F1: {best_f1:.4f}")
        
    save_checkpoint(model, optimizer, epoch, loss, f1, cp_path)
    return train_losses, f1s,test_accs, gradient_norms

