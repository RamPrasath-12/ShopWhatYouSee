import os, pandas as pd

print("Images:", len(os.listdir("data/images")))
df = pd.read_csv("data/product_database.csv")
print("CSV rows:", len(df))
