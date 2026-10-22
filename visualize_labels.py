from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "label_distribution.png"

df = pd.read_csv(DATA_FILE)
LABEL = " Label"

# --- Features vs target (we only look; df is not changed) ---
feature_names = [col for col in df.columns if col != LABEL]
print("Target column  :", repr(LABEL))
print("Number of input features:", len(feature_names))

# --- Label counts and percentages ---
counts = df[LABEL].value_counts()
percentages = df[LABEL].value_counts(normalize=True) * 100
print("\nLabel counts:")
print(counts)
print("\nLabel percentages:")
print(percentages.round(2))

# --- Check for infinite values (isnull() does not detect these) ---
numeric_part = df.select_dtypes(include="number")
inf_counts = np.isinf(numeric_part).sum()
print("\nTotal infinite values:", inf_counts.sum())
print("Columns containing infinite values:")
print(inf_counts[inf_counts > 0])

# --- Are the two 'Fwd Header Length' columns identical? ---
same = (df[" Fwd Header Length"] == df[" Fwd Header Length.1"]).all()
print("\n' Fwd Header Length' and ' Fwd Header Length.1' identical:", same)

# --- Bar chart ---
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
ax.bar_label(bars, fmt="{:,.0f}")
ax.set_title("Traffic Label Distribution")
ax.set_xlabel("Label")
ax.set_ylabel("Number of flows")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
print("\nChart saved to:", OUTPUT_FILE)
plt.show()