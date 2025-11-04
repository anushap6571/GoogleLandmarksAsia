import pandas as pd

# Load full CSV
df = pd.read_csv("train.csv")

# Choose classes you want to keep, e.g., 5 classes
num_classes = 10
images_per_class = 10  # Total images per class

# Get unique landmark_ids
unique_classes = df['landmark_id'].unique()[:num_classes]  # first N classes

# Filter for selected classes and take a subset per class
subset_rows = []
for cls in unique_classes:
    cls_rows = df[df['landmark_id'] == cls].sample(n=images_per_class, random_state=42)
    subset_rows.append(cls_rows)

subset_df = pd.concat(subset_rows).reset_index(drop=True)

# Add filename and label columns
subset_df['filename'] = subset_df['id'] + ".jpg"
subset_df['label'] = subset_df['landmark_id'].astype(str)

# Save the new CSV
subset_df.to_csv("v3-train_subset.csv", index=False)
print(f"Subset CSV created: {len(subset_df)} images across {num_classes} classes")
if __name__ == "__main__":
    pass