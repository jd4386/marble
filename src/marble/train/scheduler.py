import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.cycles       = cycles
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # linear warmup
            return [base_lr * (step / max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        # cosine decay
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return [
            base_lr * 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress))
            for base_lr in self.base_lrs
        ]
