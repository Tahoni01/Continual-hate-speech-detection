import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

class ContinualTrainer:
    def __init__(self, model, tokenizer, device='cuda', batch_size=32, lr=2e-5):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.label_map = {"hatespeech": 0, "offensive": 1, "normal": 2}
        self.num_classes = len(self.label_map)
        self.loss_fn = nn.CrossEntropyLoss()
    
    # Batch tokenization & label preparation
    def prepare_batch(self, batch):
        encodings = self.tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        labels = torch.tensor([self.label_map[x] for x in batch["label"].tolist()],
                              dtype=torch.long).to(self.device)
        return encodings, labels

    # Single training step
    def train_step(self, batch):
        self.model.train()
        encodings, labels = self.prepare_batch(batch)
        outputs = self.model(input_ids=encodings["input_ids"],
                             attention_mask=encodings["attention_mask"],
                             labels=labels)
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        preds = torch.argmax(outputs.logits, dim=1)
        return loss.item(), preds, labels

    # Train on a continual stream (list of batches)
    def train_continual(self, stream, log_every=10):
        all_losses = []
        all_preds = []
        all_labels = []

        for i, batch in enumerate(tqdm(stream, desc="Training")):
            loss, preds, labels = self.train_step(batch)
            all_losses.append(loss)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if (i + 1) % log_every == 0:
                avg_loss = sum(all_losses[-log_every:]) / log_every
                print(f"Batch {i+1}/{len(stream)} | Avg loss: {avg_loss:.4f}")

        return all_losses, all_preds, all_labels

    # Optional: evaluate without training
    def evaluate_stream(self, stream):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in stream:
                encodings, labels = self.prepare_batch(batch)
                outputs = self.model(input_ids=encodings["input_ids"],
                                     attention_mask=encodings["attention_mask"])
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        return all_preds, all_labels
