import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    """
    Clean text data for BERT/RoBERTa models
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_fake_news_dataset(input_path, output_dir):
    """
    Preprocess WELFake dataset for BERT/RoBERTa fine-tuning

    Args:
        input_path (str): Path to the input CSV file
        output_dir (str): Directory to save processed data
    """

    print("Loading dataset...")
    df = pd.read_csv(input_path, encoding='latin-1', on_bad_lines='skip')

    print(f"Original dataset shape (after skipping bad lines if any): {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nDataset Info:")
    print(df.info())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print("\nMissing values:")
    print(df.isnull().sum())
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    print("\nCleaning text data...")
    df['title_clean'] = df['title'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)
    df['combined_text'] = df['title_clean'] + ' [SEP] ' + df['text_clean']
    df = df[df['combined_text'].str.len() >= 10].copy()
    initial_count = len(df)
    df = df.drop_duplicates(subset=['combined_text'], keep='first')
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate entries")
    df['label'] = df['label'].astype(int)
    unique_labels = df['label'].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Invalid labels found: {unique_labels}. Expected only 0 and 1.")
    processed_df = df[['combined_text', 'label']].copy()
    processed_df.columns = ['text', 'label']
    processed_df['text_length'] = processed_df['text'].str.len()
    processed_df['word_count'] = processed_df['text'].str.split().str.len()
    print(f"\nProcessed dataset shape: {processed_df.shape}")
    print(f"Label distribution after preprocessing:")
    print(processed_df['label'].value_counts())
    print(f"\nText length statistics:")
    print(processed_df['text_length'].describe())
    print(f"\nWord count statistics:")
    print(processed_df['word_count'].describe())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    X = processed_df['text']
    y = processed_df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'label': y_val})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    full_output_path = os.path.join(output_dir, 'processed_full_dataset.csv')
    train_output_path = os.path.join(output_dir, 'train.csv')
    val_output_path = os.path.join(output_dir, 'validation.csv')
    test_output_path = os.path.join(output_dir, 'test.csv')
    final_full_df = processed_df[['text', 'label']].copy()
    final_full_df.to_csv(full_output_path, index=False)
    train_df.to_csv(train_output_path, index=False)
    val_df.to_csv(val_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    print(f"\nDatasets saved to:")
    print(f"Full dataset: {full_output_path}")
    print(f"Training set: {train_output_path} (samples: {len(train_df)})")
    print(f"Validation set: {val_output_path} (samples: {len(val_df)})")
    print(f"Test set: {test_output_path} (samples: {len(test_df)})")
    print(f"\nLabel distributions:")
    print(f"Training set: {train_df['label'].value_counts().to_dict()}")
    print(f"Validation set: {val_df['label'].value_counts().to_dict()}")
    print(f"Test set: {test_df['label'].value_counts().to_dict()}")
    stats = {
        'original_samples': initial_count,
        'final_samples': len(processed_df),
        'duplicates_removed': duplicates_removed,
        'avg_text_length': processed_df['text_length'].mean(),
        'avg_word_count': processed_df['word_count'].mean(),
        'label_distribution': processed_df['label'].value_counts().to_dict(),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df)
    }
    stats_df = pd.DataFrame([stats])
    stats_output_path = os.path.join(output_dir, 'preprocessing_stats.csv')
    stats_df.to_csv(stats_output_path, index=False)
    print(f"Preprocessing statistics saved to: {stats_output_path}")

    return processed_df

if __name__ == "__main__":
    input_path = "L:/Important/MCA/Mini Projec/fake_news_detection/data/raw/WELFake_Dataset.csv"
    output_dir = "L:/Important/MCA/Mini Projec/fake_news_detection/data/processed"

    try:
        processed_data = preprocess_fake_news_dataset(input_path, output_dir)
        print("\nPreprocessing completed successfully!")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        print("Please check the file path and try again.")

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
