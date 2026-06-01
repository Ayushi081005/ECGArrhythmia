import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import config

def explore_data():
    """
    Runs all data quality and visualization checks from the notebook.
    This script does not affect training; it is purely for analysis.
    """
    print("Loading data for exploration...")
    df_train = pd.read_csv(config.TRAIN_FILE, header=None)
    df_test = pd.read_csv(config.TEST_FILE, header=None)
    
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values.astype(int)
    
    print(f"Train data loaded: {df_train.shape}")
    print(f"Test data loaded: {df_test.shape}")
    
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
    
    # --- Data Quality Checks ---
    print("\n" + "="*30)
    print("Data Quality Analysis")
    print("="*30)
    print(f"Training set - Missing values: {df_train.isnull().sum().sum()}")
    print(f"Test set - Missing values: {df_test.isnull().sum().sum()}")
    print(f"Training set - Duplicate rows: {df_train.duplicated().sum()}")
    print(f"Test set - Duplicate rows: {df_test.duplicated().sum()}")
    print(f"Training set - Infinite values: {np.isinf(X_train).sum()}")
    print(f"Test set - Infinite values: {np.isinf(df_test.iloc[:, :-1].values).sum()}")

    # --- Class Distribution ---
    print("\n" + "="*30)
    print("Class Distribution (Imbalance Analysis)")
    print("="*30)
    train_counts = [np.sum(y_train == i) for i in range(5)]
    print("\nTraining Set:")
    for i in range(5):
        count = train_counts[i]
        percentage = 100 * count / len(y_train)
        print(f"  Class {i} ({class_names[i]:20s}): {count:6d} samples ({percentage:5.2f}%)")
        
    imbalance_ratio = max(train_counts) / min(train_counts)
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 10:
        print("⚠️  Dataset is HIGHLY IMBALANCED - Class weighting will be applied in train.py")

    # --- Plotly Visualizations (Interactive) ---
    print("\nGenerating interactive Plotly charts...")
    print("These will open in your web browser.")
    
    labels_dict = {
        0: "Normal",
        1: "Supraventricular Premature",
        2: "Premature Ventricular Contraction",
        3: "Fusion of Ventricular and Normal",
        4: "Unclassifiable Beat"
    }
    
    value_counts = df_train.iloc[:, -1].value_counts().sort_index().rename(labels_dict)
    
    bar_fig = px.bar(x=value_counts.index, y=value_counts.values,
                     labels={'x': 'Heartbeat Type', 'y': 'Number of Samples'},
                     text_auto=True,
                     title="Distribution of Heartbeat Types in Training Dataset",
                     color=value_counts.values,
                     color_continuous_scale='Blues')
    
    pie_fig = px.pie(names=value_counts.index, values=value_counts.values,
                     title="Percentage Distribution of Heartbeat Types in Training Dataset",
                     hole=0.3)
    
    bar_fig.update_layout(title_x=0.5, width=900, height=600, showlegend=False)
    pie_fig.update_layout(title_x=0.5, width=900, height=600)
    
    # bar_fig.show()
    # pie_fig.show()
    print("...skipping plot display in non-interactive script. Uncomment .show() to view.")

    # --- Matplotlib Sample Signals ---
    print("\nGenerating Matplotlib signal plots...")
    
    plt.figure(figsize=(15, 7))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    for i in range(5):
        class_indices = np.where(y_train == i)[0]
        idx = class_indices[5] if len(class_indices) > 5 else class_indices[0]
        plt.plot(X_train[idx], linewidth=2, color=colors[i], 
                 label=f'Class {i}: {class_names[i]}', alpha=0.8)
                 
    plt.title('ECG Signal Comparison Across All Classes', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 186)
    plt.tight_layout()
    plt.savefig('data_signal_comparison.png')
    print("Saved signal comparison plot to data_signal_comparison.png")
    # plt.show()

    print("\n" + "="*70)
    print("Data exploration complete!")
    print("="*70)

if __name__ == '__main__':
    explore_data()
