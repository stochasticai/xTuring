import os
import torch

def synthetic_data_loader(n_samples, seq_len, vocab_size):
    for i in range(n_samples):
        inp = torch.randint(0, vocab_size, (1, seq_len))
        tar = torch.rand((1, seq_len, vocab_size))
        yield inp, tar

def save_synthetic_targets(samples, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    for idx, (inp, tar) in enumerate(samples):
        target = tar
        torch.save(target, os.path.join(cache_dir, f'target_{idx}.pt'))

def load_synthetic_targets(batch_idx, batch_size, cache_dir):
    targets = []
    for idx in range(batch_idx * batch_size, (batch_idx + 1) * batch_size):
        target = torch.load(os.path.join(cache_dir, f'target_{idx}.pt'))
        targets.append(target)
    return torch.stack(targets)

n_samples = 16
seq_len = 10
vocab_size = 100
batch_size = 4
cache_dir = './synthetic_targets/'

synthetic_samples = list(synthetic_data_loader(n_samples, seq_len, vocab_size))
save_synthetic_targets(synthetic_samples, cache_dir)

for batch_idx in range(n_samples // batch_size):
    loaded_targets = load_synthetic_targets(batch_idx, batch_size, cache_dir)
    original_targets = torch.stack([tar for _, tar in synthetic_samples[batch_idx * batch_size: (batch_idx + 1) * batch_size]], dim=0)
    
    assert torch.allclose(loaded_targets, original_targets), f"Loaded targets do not match the original targets in batch {batch_idx}"
    print(f"Batch {batch_idx} targets match")
