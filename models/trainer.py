import torch
from torch.optim import AdamW
from tqdm import tqdm


class ContinualTrainer:
    def __init__(self, model, tokenizer, device="cuda",lr=2e-5):

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        # optional components
        self.replay = None
        self.ewc = None
        self.distillation = None

    # -----------------------
    # encoding batch
    # -----------------------
    def prepare_batch(self, batch, label_map):

        enc = self.tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        labels = torch.tensor(
            [label_map[x] for x in batch["label"].tolist()],
            dtype=torch.long,
            device=self.device
        )

        return enc, labels

    # -----------------------
    # SINGLE TRAIN STEP
    # -----------------------
    def train_step(self, batch, label_map):

        self.model.train()

        inputs, labels = self.prepare_batch(batch, label_map)

        outputs = self.model(**inputs)
        logits = outputs.logits

        # 1) BASE LOSS
        loss = self.ce_loss(logits, labels)

        # 2) EWC (regularization)
        if self.ewc is not None:
            loss = loss + self.ewc.ewc_loss()

        # 3) DISTILLATION
        if self.distillation is not None:

            with torch.no_grad():
                teacher_out = self.distillation["old_model"](**inputs)

            T = self.distillation.get("temperature", 2.0)
            alpha = self.distillation.get("alpha", 0.5)

            distill_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits / T, dim=-1),
                torch.nn.functional.softmax(teacher_out.logits / T, dim=-1),
                reduction="batchmean"
            ) * (T * T)

            loss = alpha * loss + (1 - alpha) * distill_loss

        # 4) BACKPROP
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5) REPLAY BUFFER UPDATE (data-level only)
        if self.replay is not None:
            self.replay.update_buffer(inputs, labels)

        preds = torch.argmax(logits, dim=1)

        return loss.item(), preds.detach().cpu(), labels.detach().cpu()

    # -----------------------
    # TRAIN LOOP
    # -----------------------
    def train_continual(self, stream, label_map, log_every=10):

        losses = []
        preds_list = []
        labels_list = []

        for i, batch in enumerate(tqdm(stream)):

            loss, preds, labels = self.train_step(batch, label_map)

            losses.append(loss)
            preds_list.append(preds)
            labels_list.append(labels)

            if (i + 1) % log_every == 0:
                print(f"[{i+1}] loss: {sum(losses[-log_every:]) / log_every:.4f}")

        return losses, preds_list, labels_list

    # -----------------------
    # EVAL
    # -----------------------
    def evaluate(self, stream, label_map):

        self.model.eval()

        preds_list, labels_list = [], []

        with torch.no_grad():
            for batch in stream:

                inputs, labels = self.prepare_batch(batch, label_map)

                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)

                preds_list.append(preds.cpu())
                labels_list.append(labels.cpu())

        return preds_list, labels_list