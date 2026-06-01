import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import config # Import our settings

class ECGDataset(Dataset):
    """Custom PyTorch Dataset for ECG signals"""
    
    def __init__(self, X, y):
        # Add channel dimension for CNN (batch, channels, length)
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(train_file, test_file, batch_size):
    """
    Loads data, applies scaling, and returns train/test DataLoaders.
    Saves the scaler to disk.
    """
    print("Loading and preprocessing data...")
    
    # --- Load Data ---
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)
    
    # Separate features (X) and labels (y)
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values.astype(int)
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values.astype(int)
    
    # --- Normalization (Scaler) ---
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using the *same* scaler
    X_test_scaled = scaler.transform(X_test)
    
    # --- SAVE THE SCALER ---
    # This is a critical step for prediction
    print(f"Saving scaler to {config.SCALER_PATH}...")
    joblib.dump(scaler, config.SCALER_PATH)
    
    # --- Create Datasets ---
    train_dataset = ECGDataset(X_train_scaled, y_train)
    test_dataset = ECGDataset(X_test_scaled, y_test)
    
    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully.")
    
    # Return y_train to calculate class weights in the train script
    return train_loader, test_loader, y_train

def get_test_data_for_eval():
    """
    Loads and scales the test data. Used by evaluate.py and predict.py
    """
    df_test = pd.read_csv(config.TEST_FILE, header=None)
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values.astype(int)
    
    # --- LOAD THE SCALER ---
    # We do NOT fit, only transform
    print(f"Loading scaler from {config.SCALER_PATH}...")
    try:
        scaler = joblib.load(config.SCALER_PATH)
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {config.SCALER_PATH}.")
        print("Please run train.py first to create and save the scaler.")
        return None, None
        
    X_test_scaled = scaler.transform(X_test)
    
    test_dataset = ECGDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Return all components needed for full evaluation
    return test_loader, X_test, y_test

if __name__ == '__main__':
    # You can run this file directly to test it
    # python preprocess.py
    print("Testing data preprocessing...")
    train_loader, test_loader, _ = get_dataloaders(
        config.TRAIN_FILE, 
        config.TEST_FILE, 
        config.BATCH_SIZE
    )
    
    # Check one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {X_batch.shape}") # Should be [batch, 1, 187]
    print(f"Batch y shape: {y_batch.shape}") # Should be [batch]