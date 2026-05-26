# Continual-hate-speech-detection# Continual Hate Speech Detection with DistilRoBERTa

This project explores **Continual Learning (CL)** for text classification in the domain of **hate speech detection**, using a pretrained transformer model (DistilRoBERTa) and multiple anti-forgetting strategies.

The goal is to simulate a **data stream setting**, where new data arrives sequentially and the model must adapt without forgetting previous knowledge.

---

## 🧠 Problem Setting

In real-world NLP applications, data is not static. Models deployed in production face:

- data distribution shifts
- new linguistic patterns
- evolving hate/offensive language
- catastrophic forgetting

This project addresses these challenges using continual learning techniques.

---

## ⚙️ Model

The base model used is:

- `DistilRoBERTa` (HuggingFace Transformers)

A classification head is added on top for:

- Hate Speech
- Offensive Language
- Normal Content

---

## 📊 Continual Learning Strategies

This project implements and compares multiple strategies:

### 🔁 1. Replay Strategy
Stores past samples in a memory buffer and replays them during training to reduce forgetting.

### 🧠 2. EWC (Elastic Weight Consolidation)
Adds a regularization term to preserve important weights based on Fisher Information.

### 🎓 3. Knowledge Distillation
Uses a previous model (teacher) to maintain past knowledge while learning new data.

---

## 🏗️ Project Structure


---

## 🚀 Training Pipeline

The training follows a **stream-based setup**:

1. Data arrives sequentially (stream)
2. Each batch is tokenized
3. Model is trained step-by-step
4. Optional continual learning strategy is applied
5. Metrics are computed per batch and globally

---

## 📈 Evaluation Metrics

- Accuracy
- Macro F1-score
- Confusion Matrix
- Per-class accuracy
- Batch-wise performance tracking

---

## 📉 Key Observations

During training, the model may exhibit:

- **catastrophic forgetting**
- **class collapse (predicting a single class)**
- instability under distribution shift

These behaviors are expected in continual learning setups without strong balancing mechanisms.

---

## 🧪 How to Run

```bash
pip install -r requirements.txt

python main.py
