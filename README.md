ECG Arrhythmia Classification using CNN-LSTM
This repository contains a PyTorch implementation for classifying Electrocardiogram (ECG) heartbeats into five distinct categories based on the MIT-BIH Arrhythmia Dataset. To capture both spatial morphological variations and temporal dependencies in the ECG signals, the project leverages a hybrid CNN-LSTM architecture.The pipeline accounts for severe dataset imbalances by computing dynamic class weights and integrating a learning rate scheduler to robustly optimize model performance.
📂 Project Structure
Based on the workspace configuration, the project layout is structured as follows:

├── mitbih_train.csv          # MIT-BIH Training Dataset

├── mitbih_test.csv           # MIT-BIH Testing/Validation Dataset

├── config.py                 # Hyperparameters, device selection, and file paths

├── preprocess.py             # Data loading, normalization, and PyTorch DataLoaders

├── model.py                  # Hybrid CNN-LSTM Neural Network architecture

├── train.py                  # Main training script with dynamic loss weighting

├── evaluate.py               # Evaluation script for computing metrics (Confusion Matrix, etc.)

├── predict.py                # Inference script for making individual predictions

├── visualize_data.py         # Script to plot and analyze raw ECG signals

├── best_ecg_model.pth        # Saved state dictionary of the best performing model

├── scaler.joblib             # Saved preprocessing/normalization parameters

├── data_signal_comparison.png# Visualization output comparing signal classes

└── prediction_example.png    # Visualization output of a model inference result

📊 Dataset & Target Classes
The model utilizes the MIT-BIH Arrhythmia Dataset, mapping the processed signals into the standard 5-class categorization defined by the AAMI:

⚙️ Key Technical Features
Hybrid Architecture: Combines Convolutional Neural Networks (CNN) for local feature extraction from signal segments with Long Short-Term Memory (LSTM) networks to process sequential structures.
Imbalance Mitigation: Real-time calculation of reciprocal class-frequency weights applied directly via nn.CrossEntropyLoss(weight=class_weights).
Adaptive Learning Rate: Configured with ReduceLROnPlateau which monitors validation loss and reduces the learning rate by a configured factor upon hitting performance plateaus.
Deterministic Configuration: Centralized parameter management through config.py for decoupled and maintainable experiments.

🚀 Getting Started1.
1.Prerequisites & Installation
Ensure you have Python 3.8+ installed. Clone the repository and install the dependencies listed in requirement.txt

cd ecg-arrhythmia
pip install -r requirement.txt

2. Dataset Setup
   Download the mitbih_train.csv and mitbih_test.csv files and place them directly into the root workspace folder as indicated in the directory layout.
3. Training the ModelRun the primary training pipeline execution script:
  python train.py
During execution, the script will output live training dynamics:
1.Automated computation of balanced class weights.
2.Per-epoch monitoring of training/validation loss and accuracy.
3.Automated checkpoints tracking the highest validating accuracy model marked by a ⭐ BEST tag.
4. Evaluation and InferenceTo evaluate the saved artifacts against alternative statistical metrics or test custom inputs:
   python evaluate.py
   python predict.py
📈 Outputs and Performance
 TrackingUpon completion, train.py automatically generates diagnostic assets mapped via your global settings:best_ecg_model.pth:
Evaluated best model weights stored safely for downstream serving.
Training Curves Chart: A dual-subplot visualization tracking Loss vs. Epoch alongside Accuracy (%) vs. Epoch saved to the path specified in config.py.

👥 Authors & AcknowledgmentsDataset Source: PhysioNet MIT-BIH Arrhythmia DatabaseDeveloped as an end-to-end deep learning framework for biomedical signal classification.
