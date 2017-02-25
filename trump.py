import pandas as pd

df = pd.read_csv("./data/trump_tweets.csv", encoding="latin-1")

X = df["Tweet_Text"]