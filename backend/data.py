# backend/pipeline/data.py
import os
import pandas as pd
import kagglehub

def load_news():
    # Download dataset from KaggleHub
    print("Downloading dataset...")
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")

    # Read train and test CSVs
    train_path = os.path.join(path, "train.csv")
    test_path  = os.path.join(path, "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Combine train and test sets
    print("Formatting dataset...")
    news = pd.concat([train, test], ignore_index=True)

    # Remove Class Index column
    if "Class Index" in news.columns:
        news = news.drop(columns=["Class Index"])

    return news
