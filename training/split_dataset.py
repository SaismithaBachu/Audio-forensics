import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load metadata (tab-separated)
df = pd.read_csv("../datasets/dcase_tau/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv", sep="\t")

# -------- First Split: Train (70%) vs Temp (30%) --------
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df["identifier"]))

train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

# -------- Second Split: Validation (15%) vs Test (15%) --------
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["identifier"]))

val_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

# Save files
train_df.to_csv("../splits/train.csv", index=False)
val_df.to_csv("../splits/val.csv", index=False)
test_df.to_csv("../splits/test.csv", index=False)

print("Split Completed")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))
