import pandas as pd
import re


def clean_text(text):

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data(file_path: str = "data/comments.csv"):

    df = pd.read_csv(file_path)

    df = df.loc[:, ["text", "sentiments"]]

    df = df.dropna(subset=["text"])

    sentiment_map = {"neutral": 0, "positive": 1, "negative": 2}
    df["sentiments"] = df["sentiments"].map(sentiment_map)

    df.reset_index()

    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["sentiments"]

    return X, y
