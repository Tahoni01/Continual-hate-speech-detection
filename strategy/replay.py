# strategies/replay.py
import torch
import random

class ReplayStrategy:
    def __init__(self, model, buffer_size=100):
        self.model = model
        self.buffer = []
        self.buffer_size = buffer_size

    def update_buffer(self, batch_inputs, batch_labels):
        for x, y in zip(batch_inputs, batch_labels):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((x, y))
            else:
                idx = random.randint(0, self.buffer_size - 1)
                self.buffer[idx] = (x, y)

    def get_replay_batch(self, batch_size=32):
        if len(self.buffer) == 0:
            return None, None
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        inputs, labels = zip(*samples)
        return torch.stack(inputs), torch.tensor(labels)

    def compute_loss(self, criterion, batch_preds, batch_labels):
        return criterion(batch_preds, batch_labels)
