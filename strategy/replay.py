import torch
import random

class ReplayStrategy:
    def __init__(self, buffer_size=200):
        self.buffer = []
        self.buffer_size = buffer_size

    def update_buffer(self, inputs, labels):
        input_ids = inputs["input_ids"].detach().cpu()
        attention_mask = inputs["attention_mask"].detach().cpu()
        labels = labels.detach().cpu()

        for i in range(len(labels)):
            sample = (input_ids[i], attention_mask[i], labels[i])

            if len(self.buffer) < self.buffer_size:
                self.buffer.append(sample)
            else:
                idx = random.randint(0, self.buffer_size - 1)
                self.buffer[idx] = sample

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None

        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        input_ids, attention_mask, labels = zip(*batch)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask)
        }, torch.tensor(labels)