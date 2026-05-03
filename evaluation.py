import time
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Times each models training and inference
class Timer:
    def __init__(self):
        self.start_ts = time.time()
    def stop(self):
        self.end_ts = time.time()
        mins, secs = divmod(self.end_ts - self.start_ts, 60)
        print(f'Time Elapsed: {mins}m {secs}s')

# Evaulate Models results
class Evaluate_model:
    def __init__(self, y_true, y_pred, modelname: str, encoders_dict: dict):
        self.y_true = y_true
        self.y_pred = y_pred
        self.modelname = modelname
        try:
            self.class_names = encoders_dict["label_encoder"]["APP"].classes_
        except (KeyError, AttributeError):
            self.class_names = None

    # Prints accuracy, macro f1, weighted f1 and classification report to console
    def get_main_metrics(self):
        acc         = accuracy_score(self.y_true, self.y_pred)
        macro_f1    = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)

        print(f"------- {self.modelname} -------")
        print(f"Accuracy    : {acc:.4f}")
        print(f"Macro F1    : {macro_f1:.4f}")
        print(f"Weighted F1 : {weighted_f1:.4f}")
        print(classification_report(self.y_true, self.y_pred, target_names=self.class_names, zero_division=0))

    # Prints confusion matrix of top n / bottom n peforming classes
    def get_confusion_matrix(self, on_top: bool, num=10):
        per_class_f1 = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        if on_top:
            idx = np.argsort(per_class_f1)[-num:][::-1]
            label = f"top {num} best"
        else:
            idx = np.argsort(per_class_f1)[:num]
            label = f"bottom {num} worst"
        cm      = confusion_matrix(self.y_true, self.y_pred, labels=idx)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        tick_labels = [self.class_names[i] for i in idx] if self.class_names is not None else idx

        fig, ax = plt.subplots(figsize=(18, 15))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{self.modelname} — Normalised Confusion Matrix ({label} performing classes by F1)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()
        print("\n")