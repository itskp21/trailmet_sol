import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vit_pytorch import ViT

# Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize images
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# Define Vision Transformer (ViT)
model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=100,
    dim=256,       # embedding dimension
    depth=6,       # number of transformer blocks
    heads=8,       # number of attention heads
    mlp_dim=512,   # dimension of feed-forward layers
    dropout=0.1,
    emb_dropout=0.1
).cuda() # Use GPU if available

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Add L1 regularization to promote sparsity
l1_lambda = 1e-4
def l1_penalty(model):
    l1_norm = sum(p.abs().sum() for p in model.parameters() if len(p.shape) > 1)
    return l1_lambda * l1_norm

# Training function with L1 regularization
def train_model(model, trainloader, optimizer, criterion, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) + l1_penalty(model)  # Add L1 regularization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")

# Function to prune model channels based on BatchNorm scaling factors (gamma)
def prune_model(model, threshold=0.05):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            mask = module.weight.abs() > threshold
            module.weight.data.mul_(mask)
            module.bias.data.mul_(mask)

# Fine-tuning after pruning
def fine_tune_model(model, trainloader, testloader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Fine-tune Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")

        # Evaluate after each epoch
        evaluate_model(model, testloader)

# Model evaluation function
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

# Train the model with L1 regularization for slimming
train_model(model, trainloader, optimizer, criterion, num_epochs=20)

# Prune the model based on BatchNorm scaling parameters
prune_model(model, threshold=0.05)

# Fine-tune the pruned model
fine_tune_model(model, trainloader, testloader, optimizer, criterion, num_epochs=10)

# Final evaluation after pruning and fine-tuning
evaluate_model(model, testloader)