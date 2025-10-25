import torch
import torch.nn.functional as F
# --- CHANGE: Import GCNConv instead of SAGEConv ---
from torch_geometric.nn import GCNConv 
from torch_geometric.data import Data
import os

print("--- Phase 4: Train GNN Model (GCN, 1000 Epochs) ---")

# 1. Define your GNN Model using GCNConv
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        # --- CHANGE: Using GCNConv layers ---
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.out = torch.nn.Linear(32, out_channels) # Output layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.out(x)
        return x

# 2. Define the main training function
def main():
    # Load the prepared graph data (now averaged over all months)
    try:
        graph_data = torch.load('./data/training_graph.pt')
    except FileNotFoundError:
        print("Error: 'training_graph.pt' not found. Run '2_prepare_graph_data.py' first.")
        exit()

    # Initialize model and optimizer
    model = GNN(in_channels=graph_data.num_node_features, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training Loop
    print("Training GNN...")
    model.train()
    # --- CHANGE: Train for 1000 epochs ---
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        
        # Calculate loss ONLY on the nodes that have real station data
        loss = F.mse_loss(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Print less frequently for longer training
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch:04d}, Loss: {loss:.4f}')

    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), 'gnn_model.pth')
    print("Success! Saved updated trained model to 'gnn_model.pth'.")
    print("You are ready for the final step.")

# 3. Run main() only when executed directly
if __name__ == "__main__":
    main()
