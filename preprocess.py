import pandas as pd
import re
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path("L:/Important/MCA/Mini Project/fake_news_detection")
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

def remove_special_characters(text):
    """Remove special characters from text, keeping only alphanumeric and spaces"""
    if pd.isna(text):
        return ""
    # Convert to string and remove special characters, keeping only letters, numbers, and spaces
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_csv_with_encoding(file_path):
    """Try multiple encodings to load CSV file"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16', 'utf-8-sig']

    for encoding in encodings:
        try:
            logger.info(f"Trying to load CSV with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            logger.info(f"Successfully loaded CSV with encoding: {encoding}")
            return df
        except UnicodeDecodeError as e:
            logger.warning(f"Failed with encoding {encoding}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Failed with encoding {encoding}: {e}")
            continue

    # If all encodings fail, try with error handling
    try:
        logger.info("Trying with error handling (ignore errors)")
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', on_bad_lines='skip')
        logger.info("Successfully loaded CSV with error handling")
        return df
    except Exception as e:
        logger.error(f"All encoding attempts failed: {e}")
        raise

def preprocess():
    # Load raw CSV with robust encoding handling
    try:
        df = load_csv_with_encoding(RAW_CSV)
    except Exception as e:
        logger.error(f"Could not load CSV file: {e}")
        raise

    logger.info(f"Raw dataset loaded: {df.shape[0]} rows, columns: {list(df.columns)}")

    # Check if required columns exist
    required_columns = ['title', 'text', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset. Available columns: {list(df.columns)}")

    logger.info(f"Original dataset shape: {df.shape}")

    # 1. Clean label column - keep only rows with '0' or '1' in label
    logger.info("Cleaning label column...")

    # Handle NaN values in label column first
    before_label_clean = len(df)
    df = df.dropna(subset=['label'])
    nan_labels_removed = before_label_clean - len(df)
    if nan_labels_removed > 0:
        logger.info(f"Removed {nan_labels_removed} rows with NaN labels")

    # Convert label to string first to handle mixed types
    df['label'] = df['label'].astype(str).str.strip()

    # Keep only rows where label is exactly '0' or '1'
    valid_labels = df['label'].isin(['0', '1'])
    invalid_count = (~valid_labels).sum()

    if invalid_count > 0:
        logger.info(f"Removing {invalid_count} rows with invalid labels")
        unique_invalid = df[~valid_labels]['label'].unique()
        logger.info(f"Invalid label values found: {unique_invalid}")

    df = df[valid_labels].copy()

    # Convert back to integer
    df['label'] = df['label'].astype(int)

    logger.info(f"After label cleaning: {df.shape[0]} rows")

    # 2. Handle NaN values in title and text columns
    logger.info("Handling NaN values in title and text columns...")

    # Fill NaN values with empty string
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')

    # Convert to string to handle any mixed types
    df['title'] = df['title'].astype(str)
    df['text'] = df['text'].astype(str)

    # 3. Remove special characters from title and text columns
    logger.info("Removing special characters from title and text columns...")

    df['title'] = df['title'].apply(remove_special_characters)
    df['text'] = df['text'].apply(remove_special_characters)

    # 4. Remove rows where title or text columns are empty
    logger.info("Removing rows with empty title or text...")

    # Count empty rows before removal
    empty_title = (df['title'].str.len() == 0) | (df['title'] == '') | (df['title'] == 'nan')
    empty_text = (df['text'].str.len() == 0) | (df['text'] == '') | (df['text'] == 'nan')
    empty_either = empty_title | empty_text

    empty_count = empty_either.sum()
    if empty_count > 0:
        logger.info(f"Removing {empty_count} rows with empty title or text")
        logger.info(f"  - Empty title: {empty_title.sum()}")
        logger.info(f"  - Empty text: {empty_text.sum()}")

    df = df[~empty_either].copy()

    logger.info(f"After removing empty rows: {df.shape[0]} rows")

    # Remove duplicates based on title and text combination
    before_dup = len(df)
    df = df.drop_duplicates(subset=['title', 'text'], keep='first')
    duplicates_removed = before_dup - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")

    # Remove rows with very short content (less than 3 characters total)
    before_short = len(df)
    short_content = ((df['title'].str.len() + df['text'].str.len()) < 3)
    df = df[~short_content].copy()
    short_removed = before_short - len(df)
    if short_removed > 0:
        logger.info(f"Removed {short_removed} rows with very short content")

    # Final dataset info
    logger.info(f"Final dataset shape: {df.shape}")

    if len(df) == 0:
        raise ValueError("No data remaining after preprocessing. Please check your input data.")

    logger.info(f"Label distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        logger.info(f"  Label {label}: {count} samples ({count/len(df)*100:.1f}%)")

    # Print statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Average title length: {df['title'].str.len().mean():.1f} characters")
    logger.info(f"Average text length: {df['text'].str.len().mean():.1f} characters")
    logger.info(f"Average title word count: {df['title'].str.split().str.len().mean():.1f} words")
    logger.info(f"Average text word count: {df['text'].str.split().str.len().mean():.1f} words")

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only the required columns in the final dataset
    final_df = df[['title', 'text', 'label']].copy()

    # Print number of rows before saving (as requested)
    print(f"\n{'='*50}")
    print(f"FINAL PREPROCESSING RESULTS")
    print(f"{'='*50}")
    print(f"Number of rows to be saved: {len(final_df)}")
    print(f"Columns: {list(final_df.columns)}")
    print(f"Label distribution: {final_df['label'].value_counts().to_dict()}")
    print(f"{'='*50}")

    # Save the processed dataset
    output_file = OUTPUT_DIR / "WELFake_preprocessed.csv"
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved preprocessed dataset to: {output_file}")

    # Also create train/validation/test splits
    from sklearn.model_selection import train_test_split

    # Check if we have enough samples for splitting
    if len(final_df) < 10:
        logger.warning("Dataset too small for train/validation/test splits. Saving only the full dataset.")
        return final_df

    # Split into train/temp (80/20)
    train_df, temp_df = train_test_split(
        final_df, test_size=0.2, random_state=42, stratify=final_df['label']
    )

    # Split temp into validation/test (10/10 from original)
    if len(temp_df) >= 4:  # Need at least 4 samples to split into val/test
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
        )
    else:
        # If temp is too small, use it all as validation
        val_df = temp_df
        test_df = temp_df.copy()

    # Save splits
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding='utf-8')
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False, encoding='utf-8')
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding='utf-8')

    logger.info(f"Saved training set: {len(train_df)} samples")
    logger.info(f"Saved validation set: {len(val_df)} samples") 
    logger.info(f"Saved test set: {len(test_df)} samples")

    return final_df

if __name__ == "__main__":
    try:
        preprocessed_data = preprocess()
        print("\nPreprocessing completed successfully!")
        print(f"Final dataset contains {len(preprocessed_data)} rows")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please check if the file 'WELFake_Dataset.csv' exists at:")
        print("L:/Important/MCA/Mini Project/fake_news_detection/data/raw/")

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
