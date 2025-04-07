import torch
import torch.nn as nn
from .cnn_model import CNNBlock
from .transformer_module import TransformerBlock

class HybridModel(nn.Module):
    def __init__(self, cnn_input_dim, num_classes, transformer_input_dim, num_heads, ff_dim):
        super(HybridModel, self).__init__()
        self.cnn = CNNBlock(cnn_input_dim, 128)
        self.transformer = TransformerBlock(transformer_input_dim, num_heads, ff_dim)
        self.classifier = nn.Linear(128 + transformer_input_dim, num_classes)

    def forward(self, cnn_input, transformer_input):
        cnn_out = self.cnn(cnn_input)  # shape: [batch, 128]
        transformer_input = transformer_input.permute(1, 0, 2)  # [seq_len, batch, embed_dim]
        transformer_out = self.transformer(transformer_input)
        transformer_out = transformer_out.mean(dim=0)  # Global average pooling: [batch, embed_dim]
        combined = torch.cat((cnn_out, transformer_out), dim=1)
        return self.classifier(combined)
# Integration of CNN and Transformer
