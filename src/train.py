import torch
import torch.nn as nn
import torch.optim as optim
from src.hybrid_framework import HybridModel

# Dummy data for testing model structure
batch_size = 4
cnn_input_dim = 100
sequence_length = 10
transformer_input_dim = 64

model = HybridModel(
    cnn_input_dim=cnn_input_dim,
    num_classes=2,
    transformer_input_dim=transformer_input_dim,
    num_heads=4,
    ff_dim=128
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Fake input data
cnn_input = torch.randn(batch_size, cnn_input_dim)
transformer_input = torch.randn(batch_size, sequence_length, transformer_input_dim)
labels = torch.randint(0, 2, (batch_size,))

# Training step
model.train()
outputs = model(cnn_input, transformer_input)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
# Training script placeholder
