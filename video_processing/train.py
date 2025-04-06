import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from pointnet_model import PointNet
from dataset_loader import SpiderDataset
from collate_fn import collate_fn

# ✅ Load datasets
train_dataset = SpiderDataset("video_processing/dataset/train")
val_dataset = SpiderDataset("video_processing/dataset/val")

# ✅ Create DataLoaders (Lower batch size to reduce memory usage)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, pin_memory=True)

# ✅ Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet().to(device)

# ✅ Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ✅ Track losses & accuracy
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# ✅ Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for data, species_target, sex_target, maturity_target in progress_bar:
        data, species_target, sex_target, maturity_target = data.to(device), species_target.to(device), sex_target.to(device), maturity_target.to(device)

        optimizer.zero_grad()
        species_out, sex_out, maturity_out = model(data)

        loss_species = criterion(species_out, species_target)
        loss_sex = criterion(sex_out, sex_target)
        loss_maturity = criterion(maturity_out, maturity_target)

        loss = loss_species + loss_sex + loss_maturity
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})

    train_losses.append(total_loss / len(train_loader))

    # ✅ Validation step
    model.eval()
    val_loss = 0
    correct_species, correct_sex, correct_maturity = 0, 0, 0
    total = 0

    with torch.no_grad():
        for data, species_target, sex_target, maturity_target in val_loader:
            data, species_target, sex_target, maturity_target = data.to(device), species_target.to(device), sex_target.to(device), maturity_target.to(device)

            species_out, sex_out, maturity_out = model(data)

            loss_species = criterion(species_out, species_target)
            loss_sex = criterion(sex_out, sex_target)
            loss_maturity = criterion(maturity_out, maturity_target)
            val_loss += (loss_species + loss_sex + loss_maturity).item()

            pred_species = species_out.argmax(dim=1)
            pred_sex = sex_out.argmax(dim=1)
            pred_maturity = maturity_out.argmax(dim=1)

            correct_species += (pred_species == species_target).sum().item()
            correct_sex += (pred_sex == sex_target).sum().item()
            correct_maturity += (pred_maturity == maturity_target).sum().item()

            total += species_target.size(0)

    val_losses.append(val_loss / len(val_loader))
    train_accuracies.append(correct_species / total)
    val_accuracies.append(correct_sex / total)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Species Acc: {train_accuracies[-1]:.2%}, Sex Acc: {val_accuracies[-1]:.2%}")

    torch.cuda.empty_cache()  # ✅ Free memory between epochs

# ✅ Save model
torch.save(model.state_dict(), "pointnet_spiderweb.pth")
print("✅ Model training complete!")

# ✅ Plot Training Progress
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_accuracies, label="Species Accuracy")
plt.plot(range(num_epochs), val_accuracies, label="Sex Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Species & Sex Accuracy")

plt.show()
