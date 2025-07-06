import pandas as pd

# Load data
df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.info())
print(df['Class'].value_counts())  # 0 = normal, 1 = fraud