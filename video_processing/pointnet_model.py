import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes_species=3, num_classes_sex=2, num_classes_maturity=2):
        super(PointNet, self).__init__()

        # ✅ Increased filter sizes (max 128)
        self.conv1 = nn.Conv1d(3, 32, 1)   # 3 → 32
        self.conv2 = nn.Conv1d(32, 64, 1)  # 32 → 64
        self.conv3 = nn.Conv1d(64, 128, 1) # 64 → 128

        self.fc1 = nn.Linear(128, 64)  # 128 → 64
        self.fc2 = nn.Linear(64, 32)   # 64 → 32

        self.species_fc = nn.Linear(32, num_classes_species)
        self.sex_fc = nn.Linear(32, num_classes_sex)
        self.maturity_fc = nn.Linear(32, num_classes_maturity)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 3 → 32
        x = F.relu(self.bn2(self.conv2(x)))  # 32 → 64
        x = F.relu(self.bn3(self.conv3(x)))  # 64 → 128

        x = torch.max(x, 2, keepdim=True)[0]  # Global max pooling
        x = x.view(-1, 128)  # Flattened feature vector

        x = F.relu(self.bn4(self.fc1(x)))  # 128 → 64
        x = F.relu(self.bn5(self.fc2(x)))  # 64 → 32

        species_out = self.species_fc(x)
        sex_out = self.sex_fc(x)
        maturity_out = self.maturity_fc(x)

        return species_out, sex_out, maturity_out
