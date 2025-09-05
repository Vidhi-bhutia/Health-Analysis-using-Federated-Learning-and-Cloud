import os
import pandas as pd
import numpy as np

# Paths
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
HOSPITALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/hospital'))

# Disease files and their extensions
DISEASE_FILES = {
    'anemia': 'anemia.csv',
    'asthma': 'asthma.csv',
    'breast_cancer': 'breast_cancer.csv',
    'diabetes': 'diabetes.csv',
    'stroke': 'stroke.csv',
}

HOSPITALS = ['Hospital A', 'Hospital B', 'Hospital C']

def ensure_hospital_dirs():
    # Create the main hospital directory if it doesn't exist
    os.makedirs(HOSPITALS_DIR, exist_ok=True)
    # Create subfolders for each hospital
    for hospital in HOSPITALS:
        path = os.path.join(HOSPITALS_DIR, hospital)
        os.makedirs(path, exist_ok=True)

# Split and save data

def clean_data(df, disease):
    # Remove empty values
    df = df.dropna(axis=0, how='any')
    # Disease-specific cleaning
    if disease == 'diabetes' and 'smoking_history' in df.columns:
        df = df[df['smoking_history'].str.lower() != 'no info']
    # For new stroke dataset: keep only relevant columns (remove any with only one unique value or >95% same value)
    if disease == 'stroke':
        # Remove columns with only one unique value
        nunique = df.nunique()
        cols_to_drop = nunique[nunique <= 1].index.tolist()
        df = df.drop(columns=cols_to_drop)
        # Remove columns with >95% same value
        for col in df.columns:
            top_freq = df[col].value_counts(normalize=True, dropna=False).max()
            if top_freq > 0.95:
                df = df.drop(columns=[col])
        return df
    # General cleaning for other datasets
    nunique = df.nunique()
    cols_to_drop = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=cols_to_drop)
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).max()
        if top_freq > 0.95:
            df = df.drop(columns=[col])
    return df

def split_and_save():
    for disease, filename in DISEASE_FILES.items():
        file_path = os.path.join(RAW_DATA_DIR, filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"Unsupported file type for {filename}")
            continue
        # Clean the data
        df = clean_data(df, disease)
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Split into 3 roughly equal parts
        splits = np.array_split(df, 3)
        for i, hospital in enumerate(HOSPITALS):
            out_path = os.path.join(HOSPITALS_DIR, hospital, f"{disease}.csv")
            splits[i].to_csv(out_path, index=False)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    ensure_hospital_dirs()
    split_and_save()
