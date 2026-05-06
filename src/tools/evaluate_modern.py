"""
Evaluate the trained modern fatigue model on the held-out test split.
The split images are face crops produced by `preprocess_videos.py`, so the
test transform mirrors what `ModernFatigueDetector.analyze` does live.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.pipelines.fatigue_modern import ModernFatigueDetector


def evaluate(threshold: float = 0.5) -> dict:
    device = torch.device("cpu")
    print(f"Evaluating on {device}...")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_data = datasets.ImageFolder('data/splits/test', test_transform)
    if len(test_data) == 0:
        raise RuntimeError("data/splits/test is empty - run preprocess_videos first")
    loader = DataLoader(test_data, batch_size=32, shuffle=False)

    config = {'modern': {'threshold': threshold}}
    detector = ModernFatigueDetector(config)
    detector.model.to(device).eval()

    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = detector.model(inputs)
            preds = (torch.sigmoid(outputs) > threshold).float()

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            tp += int(((preds == 1) & (labels == 1)).sum().item())
            fp += int(((preds == 1) & (labels == 0)).sum().item())
            tn += int(((preds == 0) & (labels == 0)).sum().item())
            fn += int(((preds == 0) & (labels == 1)).sum().item())

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("-" * 40)
    print(f"Total samples : {total}")
    print(f"Accuracy      : {100 * accuracy:.2f}%")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1            : {f1:.4f}")
    print(f"Confusion (TP/FP/TN/FN): {tp}/{fp}/{tn}/{fn}")
    print("-" * 40)

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
    }


if __name__ == "__main__":
    evaluate()
