import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture for ECG signal classification
    CNN layers: Extract spatial features from signals
    LSTM layers: Capture temporal dependencies
    """

    def __init__(self, n_classes=config.N_CLASSES, input_channels=config.INPUT_CHANNELS, input_length=config.INPUT_LENGTH):
        super(CNN_LSTM, self).__init__()
        
        # CNN blocks
        # Conv Block 1
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.2)
        
        # Calculate size after convolutions
        # After 3 MaxPool layers with kernel_size=2: input_length // (2^3)
        # Note: This calculation is illustrative. The permute and LSTM
        # will handle the sequence length dynamically.
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,  # number of CNN feature maps
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # Bidirectional LSTM for better performance
        )
        
        # Fully Connected layers
        # Bidirectional LSTM outputs 2 * hidden_size
        self.fc1 = nn.Linear(128, 64)  # 64 * 2 = 128
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # CNN part
        # x shape: (batch, 1, length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Prepare for LSTM: (batch, channels, length) -> (batch, length, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM part
        # x shape: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take last output from LSTM
        x = lstm_out[:, -1, :]
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # You can run this file directly to test it
    # python model.py
    print("Testing model definition...")
    model = CNN_LSTM()
    print(model)
    print(f'\nTotal number of parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Test a forward pass
    test_input = torch.randn(config.BATCH_SIZE, config.INPUT_CHANNELS, config.INPUT_LENGTH)
    output = model(test_input)
    print(f"\nTest output shape: {output.shape}") # Should be [batch, n_classes]
