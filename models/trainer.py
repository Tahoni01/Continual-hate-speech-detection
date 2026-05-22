# trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

class ContinualTrainer:
    def __init__(self, model, tokenizer, device='cuda', batch_size=32, lr=2e-5, strategy=None, label_map=None):
        """
        Continual Trainer per classificazione testuale con strategie di replay/CL.
        """
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.strategy = strategy

        # Label map parametrica
        self.label_map = label_map or {"hatespeech": 0, "offensive": 1, "normal": 2}
        self.num_classes = len(self.label_map)
        self.loss_fn = nn.CrossEntropyLoss()

    # ---------------------------
    # Preparazione batch
    # ---------------------------
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

    # ---------------------------
    # Single training step
    # ---------------------------
    def train_step(self, batch):
        self.model.train()
        encodings, labels = self.prepare_batch(batch)

        # Forward pass
        outputs = self.model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            labels=labels
        )

        loss = outputs.loss

        # Strategie CL (replay, EWC, ecc.)
        if self.strategy:
            if hasattr(self.strategy, "compute_loss"):
                loss = self.strategy.compute_loss(self.loss_fn, outputs.logits, labels)
            elif hasattr(self.strategy, "ewc_loss"):
                loss = loss + self.strategy.ewc_loss()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Aggiorna buffer se presente
        if hasattr(self.strategy, "update_buffer"):
            self.strategy.update_buffer(
                {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]},
                labels
            )

        preds = torch.argmax(outputs.logits, dim=1)
        loss_value = loss.item() if loss is not None else 0.0

        return loss_value, preds.cpu(), labels.cpu()

    # ---------------------------
    # Train su stream continuo
    # ---------------------------
    def train_continual(self, stream, log_every=10):
        all_losses = []
        all_preds = []
        all_labels = []

        for i, batch in enumerate(tqdm(stream, desc="Training")):
            loss, preds, labels = self.train_step(batch)
            all_losses.append(loss)
            all_preds.append(preds)
            all_labels.append(labels)

            if (i + 1) % log_every == 0 or i == len(stream)-1:
                avg_loss = sum(all_losses[max(0, i-log_every+1):i+1]) / min(log_every, i+1)
                print(f"Batch {i+1}/{len(stream)} | Avg loss: {avg_loss:.4f}")

        return all_losses, all_preds, all_labels

    # ---------------------------
    # Evaluate su stream senza training
    # ---------------------------
    def evaluate_stream(self, stream):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in stream:
                encodings, labels = self.prepare_batch(batch)
                outputs = self.model(
                    input_ids=encodings["input_ids"],
                    attention_mask=encodings["attention_mask"]
                )
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        return all_preds, all_labels
