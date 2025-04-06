import torch

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized point clouds.
    It pads point clouds to the size of the largest one in the batch.
    """
    max_points = max([item[0].shape[1] for item in batch])  # Find the max number of points

    padded_batch = []
    for data, species, sex, maturity in batch:
        pad_size = max_points - data.shape[1]
        if pad_size > 0:
            padding = torch.zeros((3, pad_size))  # Pad with zeros
            data = torch.cat([data, padding], dim=1)

        padded_batch.append((data, species, sex, maturity))

    return torch.utils.data.dataloader.default_collate(padded_batch)
