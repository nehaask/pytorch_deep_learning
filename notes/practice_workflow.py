# =========================
# Day 1 — Fundamentals & Core PyTorch
# =========================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------
# 1. Linear Regression (NumPy + PyTorch)
# -------------------------
import numpy as np

# Synthetic dataset
X_np = np.random.rand(100, 1)
y_np = 3*X_np + 2 + 0.1*np.random.randn(100,1)

# Convert to PyTorch tensors
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# Linear model
linear_model = nn.Linear(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.1)

X_device = X.to(device)
y_device = y.to(device)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = linear_model(X_device)
    loss = criterion(y_pred, y_device)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------
# 2. PyTorch Tensor & Autograd Practice
# -------------------------
x = torch.randn(5, requires_grad=True)
y = x**2 + 2*x + 1
y_sum = y.sum()
y_sum.backward()
print("Gradients:", x.grad)

# -------------------------
# 3. CNN for MNIST
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Quick training loop (1 epoch for demo)
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = cnn_model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print("Day 1 complete: fundamentals + CNN basics done!")

# =========================
# Day 2 — Advanced Topics & Debugging
# =========================

# -------------------------
# 4. Custom Dataset Example
# -------------------------
class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Example data
X_custom = torch.randn(50, 1, 28, 28)
y_custom = torch.randint(0, 10, (50,))
dataset = MyDataset(X_custom, y_custom)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -------------------------
# 5. Training Loop + Evaluation
# -------------------------
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# Use SimpleCNN on custom dataset for practice
train_model(cnn_model, loader, nn.CrossEntropyLoss(), optim.Adam(cnn_model.parameters()))

# -------------------------
# 6. Checkpointing & Resuming
# -------------------------
torch.save({
    'model_state_dict': cnn_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "checkpoint.pth")

# Load checkpoint
checkpoint = torch.load("checkpoint.pth")
cnn_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("Checkpoint loaded successfully!")

# -------------------------
# 7. Multi-GPU / DataParallel (if available)
# -------------------------
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs available, using DataParallel.")
    cnn_model = nn.DataParallel(cnn_model)
    cnn_model.to(device)
else:
    print("Single GPU or CPU, skipping DataParallel.")

# -------------------------
# 8. Overfitting / Debugging Practice
# -------------------------
# Intentionally train a small model on very few samples
tiny_loader = DataLoader(dataset, batch_size=2, shuffle=True)
train_model(cnn_model, tiny_loader, nn.CrossEntropyLoss(), optim.Adam(cnn_model.parameters()))
print("Check for overfitting behavior (loss drops quickly).")
