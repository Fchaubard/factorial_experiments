import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import numpy as np
import random
import wandb

# Initialize wandb and set configurations
wandb.init(project='ResNet18_CIFAR100_GAF', config={
    'GAF': False,  # Set to True to enable Gradient Agreement Filtering
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'weight_decay_type': 'l2',  # 'l1' or 'l2'
    'batch_size': 128,
    'num_val_iters': 100,  # Number of iterations between validations
    'epochs': 10,  # Number of training epochs
})
config = wandb.config

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# Load CIFAR-100 dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# Create a mapping from class to indices for sampling
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_dataset):
    class_indices[label].append(idx)

# Function to sample IID minibatches for standard training
def sample_iid_mbs(full_dataset, class_indices, batch_size):
    num_classes = len(class_indices)
    samples_per_class = batch_size // num_classes
    batch_indices = []
    for cls in class_indices:
        indices = random.sample(class_indices[cls], samples_per_class)
        batch_indices.extend(indices)
    # If batch_size is not divisible by num_classes, fill the rest randomly
    remaining = batch_size - len(batch_indices)
    if remaining > 0:
        all_indices = [idx for idx in range(len(full_dataset))]
        batch_indices.extend(random.sample(all_indices, remaining))
    # Create a Subset
    batch = Subset(full_dataset, batch_indices)
    return batch

# Function to sample IID minibatches for GAF
def sample_iid_mbs_for_GAF(full_dataset, class_indices):
    batch_indices1 = []
    batch_indices2 = []
    for cls in class_indices:
        # Ensure there are at least two samples per class
        if len(class_indices[cls]) >= 2:
            indices = random.sample(class_indices[cls], 2)
        else:
            indices = class_indices[cls] * 2  # Duplicate if only one sample
        batch_indices1.append(indices[0])
        batch_indices2.append(indices[1])
    # Create Subsets
    batch1 = Subset(full_dataset, batch_indices1)
    batch2 = Subset(full_dataset, batch_indices2)
    return batch1, batch2

# Gradient Agreement Filtering function (GAF)
def filter_gradients(G1, G2, epsilon=1e-1):
    filtered_grad = []
    masked = []
    total = []
    for g1, g2 in zip(G1, G2):
        agree = torch.sign(g1) == torch.sign(g2)  # Direction agreement
        similar = torch.abs(g1 - g2) < epsilon    # Magnitude similarity
        mask = agree & similar                    # Both conditions satisfied
        filtered_grad.append(mask.float() * (g1 + g2) / 2)  # Average gradients
        masked.append(torch.sum(mask.float()))
        total.append(torch.numel(mask))
    gaf_percentage = (sum(masked) / sum(total)).item() * 100
    wandb.log({'GAF_Agreement_Percentage': gaf_percentage})
    print(f"Gradient Agreement Percentage: {gaf_percentage:.2f}%")
    return filtered_grad

# Initialize the model
model = models.resnet18(num_classes=100)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer setup
if config.weight_decay_type == 'l2':
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
elif config.weight_decay_type == 'l1':
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0)
    l1_lambda = config.weight_decay
else:
    raise ValueError("weight_decay_type must be 'l1' or 'l2'")

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy_top1 = correct_top1 / total
    return avg_loss, accuracy_top1

# Test DataLoader
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

# Training loop
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    total = 0
    iteration = 0

    # Calculate total iterations per epoch
    if config.GAF:
        iterations_per_epoch = len(train_dataset) // (len(class_indices) * 2)
    else:
        iterations_per_epoch = len(train_dataset) // config.batch_size

    while iteration < iterations_per_epoch:
        if config.GAF:
            # Sample two minibatches for GAF
            batch1, batch2 = sample_iid_mbs_for_GAF(train_dataset, class_indices)
            loader1 = DataLoader(batch1, batch_size=len(batch1), shuffle=False)
            loader2 = DataLoader(batch2, batch_size=len(batch2), shuffle=False)
            data1 = next(iter(loader1))
            data2 = next(iter(loader2))
            images1, labels1 = data1[0].to(device), data1[1].to(device)
            images2, labels2 = data2[0].to(device), data2[1].to(device)

            # Forward and backward passes for first batch
            optimizer.zero_grad()
            outputs1 = model(images1)
            loss1 = criterion(outputs1, labels1)
            loss1.backward()
            G1 = [p.grad.clone() for p in model.parameters()]
            optimizer.zero_grad()

            # Forward and backward passes for second batch
            outputs2 = model(images2)
            loss2 = criterion(outputs2, labels2)
            loss2.backward()
            G2 = [p.grad.clone() for p in model.parameters()]

            # Filter gradients
            filtered_grad = filter_gradients(G1, G2, epsilon=1e-1)

            # Apply filtered gradients
            with torch.no_grad():
                for param, grad in zip(model.parameters(), filtered_grad):
                    param.grad = grad

            # L1 regularization if specified
            if config.weight_decay_type == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = (loss1 + loss2) / 2 + l1_lambda * l1_norm
            else:
                loss = (loss1 + loss2) / 2

            optimizer.step()

            # Update metrics
            running_loss += loss.item() * (images1.size(0) + images2.size(0)) / 2
            _, predicted = torch.max(outputs1.data, 1)
            total += labels1.size(0)
            correct_top1 += (predicted == labels1).sum().item()
        else:
            # Sample a minibatch for standard training
            batch = sample_iid_mbs(train_dataset, class_indices, config.batch_size)
            loader = DataLoader(batch, batch_size=len(batch), shuffle=False)
            data = next(iter(loader))
            images, labels = data[0].to(device), data[1].to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # L1 regularization if specified
            if config.weight_decay_type == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

        iteration += 1

        # Perform validation every num_val_iters iterations
        if iteration % config.num_val_iters == 0:
            train_loss = running_loss / total
            train_accuracy = correct_top1 / total
            val_loss, val_accuracy = evaluate(model, test_loader, device)
            # Log metrics to wandb
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'epoch': epoch,
                'iteration': iteration,
            })
            print(f"Epoch [{epoch+1}/{config.epochs}], Iteration [{iteration}], "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%")
            # Reset running metrics
            running_loss = 0.0
            correct_top1 = 0
            total = 0

    # End of epoch validation
    val_loss, val_accuracy = evaluate(model, test_loader, device)
    wandb.log({
        'epoch': epoch,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    })
    print(f"End of Epoch [{epoch+1}/{config.epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%")
