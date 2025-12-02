import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_predictions: bool = True
) -> Dict:
   
   # Run model inference and collect predictions

    model.eval()

    all_predictions = []
    all_labels = []
    all_confidences = []
    running_loss = 0.0
    total = 0

    logger.info("Running model evaluation...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)

            # loss
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

            # predictions
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)

            # collect results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            total += labels.size(0)

    # calculate average loss
    avg_loss = running_loss / total if criterion is not None else None

    results = {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'confidences': np.array(all_confidences),
        'loss': avg_loss,
        'total_samples': total
    }

    logger.info(f"Evaluation complete: {total} samples processed")

    return results


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int] = None
) -> Dict:
    # Calculate comprehensive classification metrics

    if num_classes is None:
        num_classes = max(labels.max(), predictions.max()) + 1

    # overall accuracy
    accuracy = accuracy_score(labels, predictions) * 100

    # precision, recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro * 100,
        'recall_macro': recall_macro * 100,
        'f1_macro': f1_macro * 100,
        'precision_weighted': precision_weighted * 100,
        'recall_weighted': recall_weighted * 100,
        'f1_weighted': f1_weighted * 100,
        'precision_per_class': precision_per_class * 100,
        'recall_per_class': recall_per_class * 100,
        'f1_per_class': f1_per_class * 100,
        'support_per_class': support_per_class,
        'num_classes': num_classes
    }

    return metrics


def calculate_top_k_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    k: int = 5
) -> float:

    # calculate Top-K accuracy (useful for 50 classes)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Computing Top-{k} Accuracy"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, top_k_pred = outputs.topk(k, dim=1, largest=True, sorted=True)

            # check if true label is in top-k predictions
            correct += sum([label.item() in top_k_pred[i].tolist()
                           for i, label in enumerate(labels)])
            total += labels.size(0)

    top_k_accuracy = 100.0 * correct / total
    return top_k_accuracy


def generate_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int] = None
) -> np.ndarray:
    
    # Generate confusion matrix

    if num_classes is None:
        num_classes = max(labels.max(), predictions.max()) + 1

    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    return cm


def get_per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:

    # Get detailed per class metrics breakdown

    num_classes = max(labels.max(), predictions.max()) + 1

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # per class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    # per class accuracy
    per_class_acc = []
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == i).sum() / mask.sum()
        else:
            class_acc = 0.0
        per_class_acc.append(class_acc)

    per_class_metrics = {}
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
        per_class_metrics[class_name] = {
            'accuracy': per_class_acc[i] * 100,
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1': f1[i] * 100,
            'support': int(support[i])
        }

    return per_class_metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 18),
    normalize: bool = False
):

    # Plot confusion matrix as heatmap

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping visualization.")
        return

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]

    # for 50 classes, show abbreviated names if too long
    if len(class_names) == 50 and len(class_names[0]) > 8:
        class_names = [name[:8] + "..." if len(name) > 8 else name
                      for name in class_names]

    plt.figure(figsize=figsize)

    # use smaller font for 50 classes
    annot_fontsize = 6 if num_classes >= 40 else 8

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        annot_kws={'size': annot_fontsize}
    )

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")

    plt.close()


def plot_per_class_metrics(
    per_class_metrics: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    top_n: Optional[int] = None
):

    # plot per class metrics as bar charts

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return

    class_names = list(per_class_metrics.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']

    # sort by support if top n specified
    if top_n is not None:
        sorted_classes = sorted(
            per_class_metrics.items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )[:top_n]
        class_names = [c[0] for c in sorted_classes]
        per_class_metrics = {c[0]: c[1] for c in sorted_classes}

    # prepare data
    data = {metric: [per_class_metrics[cls][metric] for cls in class_names]
            for metric in metrics_to_plot}

    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=figsize)

    # plot bars
    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - 1.5)
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to: {save_path}")

    plt.close()


def plot_prediction_examples(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    save_path: Optional[str] = None,
    num_examples: int = 16,
    show_correct: bool = True,
    show_incorrect: bool = True
):

    # Visualize prediction examples correct and/or misclassified

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return

    model.eval()

    correct_examples = []
    incorrect_examples = []

    with torch.no_grad():
        for images, labels in dataloader:
            images_device = images.to(device)
            labels_device = labels.to(device)

            outputs = model(images_device)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(1)

            # separate correct and incorrect
            for i in range(len(labels)):
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predictions[i].item()
                conf = confidences[i].item()

                if true_label == pred_label and show_correct:
                    correct_examples.append((img, true_label, pred_label, conf))
                elif true_label != pred_label and show_incorrect:
                    incorrect_examples.append((img, true_label, pred_label, conf))

                # stop when we have enough
                if len(correct_examples) + len(incorrect_examples) >= num_examples:
                    break

            if len(correct_examples) + len(incorrect_examples) >= num_examples:
                break

    # combine examples
    examples = []
    if show_correct and show_incorrect:
        half = num_examples // 2
        examples = correct_examples[:half] + incorrect_examples[:half]
    elif show_correct:
        examples = correct_examples[:num_examples]
    elif show_incorrect:
        examples = incorrect_examples[:num_examples]

    if len(examples) == 0:
        logger.warning("No examples found to visualize")
        return

    # plot
    n_cols = 4
    n_rows = (len(examples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    # denormalize images ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, (img, true_label, pred_label, conf) in enumerate(examples):
        # denormalize
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img_np = img.permute(1, 2, 0).numpy()

        axes[idx].imshow(img_np)
        axes[idx].axis('off')

        # title
        true_name = class_names[true_label] if true_label < len(class_names) else f"C{true_label}"
        pred_name = class_names[pred_label] if pred_label < len(class_names) else f"C{pred_label}"

        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'

        title = f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.2f}"
        axes[idx].set_title(title, fontsize=8, color=color)

    # hide empty subplots
    for idx in range(len(examples), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction examples saved to: {save_path}")

    plt.close()


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def save_metrics_json(metrics: Dict, save_path: str):

    # Save metrics to JSON file

    # convert numpy arrays to lists for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            metrics_serializable[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    logger.info(f"Metrics saved to: {save_path}")


def save_metrics_csv(per_class_metrics: Dict, save_path: str):
    # Save per class metrics to CSV file

    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed. Cannot save CSV.")
        return

    # convert to DataFrame
    df = pd.DataFrame.from_dict(per_class_metrics, orient='index')
    df.index.name = 'class'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)

    logger.info(f"Per-class metrics saved to: {save_path}")


# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class Evaluator:

    # Comprehensive evaluator for ResNeXt-50 model

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: Optional[List[str]] = None
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_full(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_top5: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict:
        # Run full evaluation with all metrics

        logger.info("=" * 80)
        logger.info("Running Full Evaluation")
        logger.info("=" * 80)

        # run inference
        eval_results = evaluate(
            self.model,
            dataloader,
            self.device,
            self.criterion,
            return_predictions=True
        )

        predictions = eval_results['predictions']
        labels = eval_results['labels']
        confidences = eval_results['confidences']

        # calculate metrics
        metrics = calculate_metrics(predictions, labels)

        # top-5 accuracy
        if compute_top5:
            top5_acc = calculate_top_k_accuracy(self.model, dataloader, self.device, k=5)
            metrics['top5_accuracy'] = top5_acc

        # confusion matrix
        cm = generate_confusion_matrix(predictions, labels)

        # per-class metrics
        per_class_metrics = get_per_class_metrics(
            predictions,
            labels,
            self.class_names
        )

        # compile full results
        full_results = {
            **metrics,
            'loss': eval_results['loss'],
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'avg_confidence': confidences.mean(),
            'total_samples': eval_results['total_samples']
        }

        # print summary
        self.print_evaluation_summary(full_results)

        # save results
        if save_dir:
            self.save_results(full_results, dataloader, save_dir)

        return full_results

    def print_evaluation_summary(self, results: Dict):
        """Print evaluation summary to console"""
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Samples: {results['total_samples']}")
        logger.info(f"Loss: {results['loss']:.4f}")
        logger.info("-" * 80)
        logger.info(f"Accuracy: {results['accuracy']:.2f}%")
        if 'top5_accuracy' in results:
            logger.info(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
        logger.info("-" * 80)
        logger.info(f"Precision (Macro): {results['precision_macro']:.2f}%")
        logger.info(f"Recall (Macro): {results['recall_macro']:.2f}%")
        logger.info(f"F1 Score (Macro): {results['f1_macro']:.2f}%")
        logger.info("-" * 80)
        logger.info(f"Precision (Weighted): {results['precision_weighted']:.2f}%")
        logger.info(f"Recall (Weighted): {results['recall_weighted']:.2f}%")
        logger.info(f"F1 Score (Weighted): {results['f1_weighted']:.2f}%")
        logger.info("-" * 80)
        logger.info(f"Average Confidence: {results['avg_confidence']:.4f}")
        logger.info("=" * 80)

    def save_results(
        self,
        results: Dict,
        dataloader: torch.utils.data.DataLoader,
        save_dir: str
    ):
        """
        Save all evaluation results
        """
        from config.config import LoggingConfig

        os.makedirs(save_dir, exist_ok=True)

        # save metrics JSON always save
        metrics_path = os.path.join(save_dir, 'metrics.json')
        save_metrics_json(results, metrics_path)

        # save per class metrics CSV always save
        csv_path = os.path.join(save_dir, 'per_class_metrics.csv')
        save_metrics_csv(results['per_class_metrics'], csv_path)

        # PNG/Plot saving
        if LoggingConfig.SAVE_CONFUSION_MATRIX_PNG:
            # plot confusion matrix
            cm_path = os.path.join(save_dir, 'confusion_matrix.png')
            plot_confusion_matrix(
                results['confusion_matrix'],
                self.class_names,
                cm_path,
                normalize=False
            )

            # plot normalized confusion matrix
            cm_norm_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
            plot_confusion_matrix(
                results['confusion_matrix'],
                self.class_names,
                cm_norm_path,
                normalize=True
            )

        if LoggingConfig.SAVE_PLOTS:
            # plot per-class metrics
            metrics_plot_path = os.path.join(save_dir, 'per_class_metrics.png')
            plot_per_class_metrics(
                results['per_class_metrics'],
                metrics_plot_path,
                top_n=30  # show top 30 classes
            )

            # plot prediction examples
            if self.class_names:
                examples_path = os.path.join(save_dir, 'prediction_examples.png')
                plot_prediction_examples(
                    self.model,
                    dataloader,
                    self.device,
                    self.class_names,
                    examples_path,
                    num_examples=16
                )

        logger.info(f"Results saved to: {save_dir} (PNG saving disabled)")

