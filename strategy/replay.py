# strategy/replay.py
import torch
import random
from strategy.base import BaseStrategy

class ReplayStrategy(BaseStrategy):
    def __init__(self, buffer_size=200):
        self.buffer      = []
        self.buffer_size = buffer_size

    # chiamato dal trainer ad ogni step — aggiorna il buffer
    def update_buffer(self, inputs, labels):
        input_ids      = inputs["input_ids"].detach().cpu()
        attention_mask = inputs["attention_mask"].detach().cpu()
        labels         = labels.detach().cpu()

        for i in range(len(labels)):
            sample = (input_ids[i], attention_mask[i], labels[i])
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(sample)
            else:
                self.buffer[random.randint(0, self.buffer_size - 1)] = sample

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None

        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        input_ids, attention_mask, labels = zip(*batch)

        max_len = max(t.shape[0] for t in input_ids)

        def pad(tensors, pad_value=0):
            return torch.stack([
                torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_value)
                for t in tensors
            ])

        return {
            "input_ids":      pad(input_ids,      pad_value=1),
            "attention_mask": pad(attention_mask, pad_value=0)
        }, torch.stack(labels)

    # replay non aggiunge loss diretta — la loss viene calcolata nel trainer
    def compute_loss(self, trainer, inputs, labels, logits):
        return 0.0

    def __len__(self):
        return len(self.buffer)