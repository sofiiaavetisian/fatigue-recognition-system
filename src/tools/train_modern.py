import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

def train_model():
    # 1. Setup Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. Data Transformations (Must match the inference script)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(), # Helps AI generalize better
            transforms.RandomRotation(10),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. Load Datasets from data/splits
    image_datasets = {x: datasets.ImageFolder(f'data/splits/{x}', data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in ['train', 'val']}
    
    # 4. Initialize Model (MobileNetV3)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 1) # Binary output (Fatigue vs Alert)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss() # Good for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])
            print(f'Epoch {epoch}/{EPOCHS-1} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 6. Save the Brain
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/fatigue_model.pt')
    print("Training finished! Model saved to models/fatigue_model.pt")

if __name__ == "__main__":
    train_model()