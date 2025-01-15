import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Step 1: Load the Pretrained AlexNet Model
def get_modified_alexnet(num_classes):
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)
    return alexnet

# Step 2: Prepare Dataset
def prepare_datasets(data_path, batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=10):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
        scheduler.step()

# Step 4: Evaluate the Model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print details for each prediction
            for i in range(len(labels)):
                expected_class = labels[i].item()
                predicted_class = predicted[i].item()
                is_correct = "Correct" if predicted_class == expected_class else "Wrong"
                print(f"Expected: {expected_class}, Predicted: {predicted_class}, {is_correct}")

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Step 5: Save the Model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Main Function
if __name__ == "__main__":
    # Parameters
    data_path = "my_photos"  # Replace with the path to your dataset
    num_classes = 4  # Number of new classes
    batch_size = 32
    epochs = 90
    learning_rate = 0.001
    model_save_path = "alexnet_custom_classes.pth"
    
    # Device Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    
    # Step 1: Modify AlexNet
    alexnet = get_modified_alexnet(num_classes)
    
    # Step 2: Load Data
    train_loader, test_loader = prepare_datasets(data_path, batch_size)
    
    # Step 3: Set Training Components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Step 4: Train the Model
    train_model(alexnet, train_loader, criterion, optimizer, scheduler, device, epochs)
    
    # Step 5: Evaluate the Model
    evaluate_model(alexnet, test_loader, device)
    
    # Step 6: Save the Model
    save_model(alexnet, model_save_path)

