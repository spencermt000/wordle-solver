import pandas as pd
from scripts.vocab import WordVocab

def load_answer_vocab(csv_path: str) -> WordVocab:
    """
    Load only the official Wordle answers from the CSV.
    Keeps rows where 'day' is not null, and returns a WordVocab.
    """
    df = pd.read_csv(csv_path)
    answer_df = df[df["day"].notna()].copy()
    answer_words = answer_df["word"].tolist()
    return WordVocab(answer_words)