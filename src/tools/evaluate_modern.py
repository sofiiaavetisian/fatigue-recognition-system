import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.pipelines.fatigue_modern import ModernFatigueDetector

def evaluate():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on {DEVICE}...")

    # 1. Load the Test Dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_data = datasets.ImageFolder('data/splits/test', test_transform)
    loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 2. Load our Trained Brain
    config = {'modern': {'threshold': 0.5}}
    detector = ModernFatigueDetector(config)
    detector.model.to(DEVICE)
    detector.model.eval()

    # 3. Run the "Exam"
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            
            # Get raw output
            outputs = detector.model(inputs)
            # Convert to probability and then to 0 or 1
            preds = torch.sigmoid(outputs) > 0.5
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print("-" * 30)
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()