from pathlib import Path
import pandas as pd

# Build the path to the CSV file, starting from this script's location
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Load the CSV into a DataFrame (a table, like an Excel sheet)
df = pd.read_csv(DATA_FILE)
print("File loaded successfully!\n")

# 1. How many rows and columns?
print("Shape (rows, columns):", df.shape)

# 2. Look at the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 3. List all column names
print("\nColumn names:")
print(list(df.columns))