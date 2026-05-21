import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy over time
plt.plot(batches, accuracies, label="Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.title("Accuracy over time")
plt.legend()
plt.show()

# Forgetting
plt.plot(batches, forgetting_scores, label="Forgetting")
plt.xlabel("Batch")
plt.ylabel("Forgetting")
plt.title("Forgetting over batches")
plt.legend()
plt.show()

# Confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
