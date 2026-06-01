import torch
import torch.nn.functional as F
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Import from our other .py files
import config
from model import CNN_LSTM

def predict_ecg(model, signal, scaler, device):
    """
    Prediction on a single ECG signal
    """
    model.eval()
    
    # 1. Normalize signal
    # Reshape to (1, 187) because scaler expects 2D array
    try:
        signal_scaled = scaler.transform(signal.reshape(1, -1))
    except ValueError as e:
        print(f"Error scaling signal: {e}")
        print("Ensure the input signal has 187 features.")
        return None, None
    
    # 2. Convert to tensor
    # Reshape to (1, 1, 187) for the model [batch, channels, length]
    signal_tensor = torch.FloatTensor(signal_scaled).unsqueeze(1).to(device)
    
    # 3. Predict
    with torch.no_grad():
        output = model(signal_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities.cpu().numpy()[0]

def main():
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)
    
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
    
    # 1. Load Model
    print(f"Loading model from {config.MODEL_PATH}...")
    model = CNN_LSTM().to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_PATH}.")
        print("Please run train.py first to create and save the model.")
        return
        
    # 2. Load Scaler
    print(f"Loading scaler from {config.SCALER_PATH}...")
    try:
        scaler = joblib.load(config.SCALER_PATH)
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {config.SCALER_PATH}.")
        print("Please run train.py first to create and save the scaler.")
        return

    # 3. Get a sample signal from the test set
    sample_idx = 10 # Let's pick a different sample
    df_test = pd.read_csv(config.TEST_FILE, header=None)
    
    # Get the *raw* signal, not the scaled one
    sample_signal = df_test.iloc[sample_idx, :-1].values
    true_label = int(df_test.iloc[sample_idx, -1])

    # 4. Make prediction
    pred_class, probs = predict_ecg(model, sample_signal, scaler, config.DEVICE)
    
    if pred_class is None:
        return # Error during prediction

    # 5. Show results
    print(f"\n{'='*70}")
    print(f"Sample Index: {sample_idx}")
    print(f"True Class: {true_label} ({class_names[true_label]})")
    print(f"Predicted Class: {pred_class} ({class_names[pred_class]})")
    print(f"Prediction Status: {'✅ CORRECT' if pred_class == true_label else '❌ INCORRECT'}")
    print(f"\nProbability Distribution:")
    for i, prob in enumerate(probs):
        marker = "👉" if i == pred_class else "  "
        print(f"{marker} {class_names[i]:30s}: {prob*100:6.2f}%")
    print(f"{'='*70}")
    
    # 6. Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Top plot - ECG Signal
    ax1.plot(sample_signal, linewidth=2, color='darkblue')
    ax1.set_title(f'ECG Signal Sample #{sample_idx}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 186)
    
    textstr = f'True Class: {true_label} ({class_names[true_label]})\nPredicted: {pred_class} ({class_names[pred_class]})'
    props = dict(boxstyle='round', facecolor='wheat' if pred_class == true_label else 'lightcoral', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
             
    # Bottom plot - Prediction Probabilities
    colors_bar = ['green' if i == pred_class else 'steelblue' for i in range(5)]
    bars = ax2.bar(range(5), probs * 100, color=colors_bar, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Probability (%)', fontsize=12)
    ax2.set_title('Model Confidence for Each Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels([f'{i}\n{class_names[i]}' for i in range(5)], rotation=0)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{prob*100:.2f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('prediction_example.png')
    print("\nSaved prediction visualization to prediction_example.png")
    # plt.show() # Uncomment if running interactively

if __name__ == '__main__':
    main()
