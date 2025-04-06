import json

# Input and output JSON files
INPUT_FILES = ["train_labels.json", "val_labels.json"]

def clean_filenames(json_file):
    """Remove file path prefixes from JSON keys."""
    with open(json_file, "r") as f:
        data = json.load(f)

    cleaned_data = {filename.split("/")[-1]: label for filename, label in data.items()}

    # Save cleaned JSON
    output_file = json_file.replace(".json", "_cleaned.json")
    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"✅ Cleaned {json_file}, saved as {output_file}")

# Process both train and validation label files
for file in INPUT_FILES:
    clean_filenames(file)
