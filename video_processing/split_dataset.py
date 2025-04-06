import json
import random

# Load the complete dataset
with open("video_processing/labels.json", "r") as f:
    data = json.load(f)

# Shuffle the dataset for randomness
items = list(data.items())
random.shuffle(items)

# Split 80% for training, 20% for validation
split_idx = int(0.8 * len(items))
train_data = dict(items[:split_idx])
val_data = dict(items[split_idx:])

# Save training data
with open("train_labels.json", "w") as f:
    json.dump(train_data, f, indent=4)

# Save validation data
with open("val_labels.json", "w") as f:
    json.dump(val_data, f, indent=4)

print(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples")
