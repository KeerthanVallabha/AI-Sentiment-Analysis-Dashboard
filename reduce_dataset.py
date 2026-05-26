import pandas as pd

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

small = df.head(50000)

small.to_csv(
    "small_dataset.csv",
    index=False,
    header=False
)

print("Created small_dataset.csv")