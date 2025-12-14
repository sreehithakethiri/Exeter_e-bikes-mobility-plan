import pandas as pd
import glob

# Folder containing  multiple CSV files
path = "data/*.csv"

# List of all CSV files
files = glob.glob(path)

print("Found files:", files)

df_list = []
for file in files:
    print("Reading:", file)
    df_list.append(pd.read_csv(file))

# Combine them  all
combined = pd.concat(df_list, ignore_index=True)


combined.to_csv("data/london_bike_hire.csv", index=False)
print("Saved combined CSV to data/london_bike_hire.csv")
