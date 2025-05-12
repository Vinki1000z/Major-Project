"""
Model definition module.
Defines the TemporalFusionTransformer model using LSTM and Multihead Attention.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Define the Temporal Fusion Transformer model class
class TemporalFusionTransformer(pl.LightningModule):
    """
    Temporal Fusion Transformer-like model using LSTM and Multihead Attention.
    Predicts the next closing price based on past sequences.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Multihead attention layer to capture dependencies
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        # Fully connected layer for output prediction
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Final output layer using last time step's output
        out = self.fc(attn_out[:, -1, :])
        return out.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Calculate mean squared error loss
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        # Print loss every 10 batches
        if batch_idx % 10 == 0:
            print(f"Train step {batch_idx}, Loss: {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Calculate validation loss
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        # Configure Adam optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]
