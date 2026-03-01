# tools/inspect_products.py
import pandas as pd
df = pd.read_csv("static/products/products_final.csv")
print(df.columns.tolist())
print(df.head())
