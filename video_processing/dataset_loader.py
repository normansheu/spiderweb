import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset

class SpiderDataset(Dataset):
    def __init__(self, dataset_folder):
        self.data_files = [f for f in os.listdir(dataset_folder) if f.endswith(".npy")]
        self.dataset_folder = dataset_folder

        self.species_map = {"N. digna": 0, "M. dana": 1, "N. litigiosa": 2}
        self.sex_map = {"Male": 0, "Female": 1}
        self.maturity_map = {"Juvenile": 0, "Mature": 1}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        point_cloud = np.load(os.path.join(self.dataset_folder, file_name))
        label_file = file_name.replace(".npy", ".json")

        with open(os.path.join(self.dataset_folder, label_file), "r") as f:
            labels = json.load(f)

        species = self.species_map.get(labels["species"], -1)
        sex = self.sex_map.get(labels["sex"], -1)
        maturity = self.maturity_map.get(labels["maturity"], -1)

        return (
            torch.tensor(point_cloud, dtype=torch.float32).T,  # Shape [3, num_points]
            torch.tensor(species, dtype=torch.long),
            torch.tensor(sex, dtype=torch.long),
            torch.tensor(maturity, dtype=torch.long)
        )
