import time
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

def get_cached_targets_seq(batch_idx, batch_size, cache_dir):
    targets = []
    for idx in range(batch_idx * batch_size, (batch_idx + 1) * batch_size):
        target = torch.load(os.path.join(cache_dir, f'target_{idx}.pt'))
        targets.append(target)
    return torch.cat(targets, dim=0)

def get_cached_targets_thread(batch_idx, batch_size, cache_dir):
    with ThreadPoolExecutor() as executor:
        targets = list(executor.map(lambda idx: torch.load(os.path.join(cache_dir, f'target_{idx}.pt')),
                                    range(batch_idx * batch_size, (batch_idx + 1) * batch_size)))
    return torch.cat(targets, dim=0)

def load_target(idx, cache_dir):
    return torch.load(os.path.join(cache_dir, f'target_{idx}.pt'))

def get_cached_targets_joblib(batch_idx, batch_size, cache_dir):
    targets = Parallel(n_jobs=-1)(delayed(load_target)(idx, cache_dir) 
                                  for idx in range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
    return torch.cat(targets, dim=0)

batch_idx = 0
batch_size = 8
cache_dir = '../val_cache'

start = time.time()
get_cached_targets_seq(batch_idx, batch_size, cache_dir)
end = time.time()
print(f"Sequential loading time: {end - start} seconds")

start = time.time()
get_cached_targets_thread(batch_idx, batch_size, cache_dir)
end = time.time()
print(f"ThreadPoolExecutor loading time: {end - start} seconds")

start = time.time()
get_cached_targets_joblib(batch_idx, batch_size, cache_dir)
end = time.time()
print(f"Joblib loading time: {end - start} seconds")