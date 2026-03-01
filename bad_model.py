import torch
import torch.nn as nn
import torch.nn.functional as F

class ResearchTransformer(nn.Module):
    """
    A transformer model with deliberately introduced deep-learning anti-patterns.
    Perfect for triggering the NeuroTidy GitHub PR Review bot!
    """
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        # 1. Standard Self-Attention Module
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, 2048)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(2048, d_model)
        
        # Output classification head
        self.fc = nn.Linear(d_model, 100)

    def forward(self, x):
        # ERROR 1: Post-Attention Normalization Breach
        # The Add & Norm layer is missing LayerNorm, leading to gradient explosion.
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out 
        
        # ERROR 2: Dimension Flattening Bug
        # Hardcoding the shape logic is very brittle and will crack with varying batch sizes.
        x = x.view(-1, 150528) 

        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        
        # ERROR 3: Loss Function Mismatch
        # Terminal sigmoid activation right before standard CrossEntropyLoss (requires logits).
        return torch.sigmoid(self.fc(x))

def train_model():
    model = ResearchTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Dummy data
    data = torch.randn(32, 512, 512)
    target = torch.randint(0, 100, (32,))

    # ERROR 4: Catastrophic Gradient Accumulation
    # Training loop missing optimizer.zero_grad(), leading to runaway gradients.
    for epoch in range(10):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss {loss.item()}")

if __name__ == "__main__":
    train_model()
