# main.py
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import torch
import osmnx as ox
import networkx as nx
from torch_geometric.data import Data
import os
import geopandas as gpd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Import or Define GNN Class ---
# Make sure this matches your trained model (GCNConv or SAGEConv)
try:
    from train_model import GNN # Assumes 3_train_model.py is train_model.py
except ImportError:
    try:
        from three_train_model import GNN
    except ImportError:
        print("Could not import GNN class, defining it locally.")
        import torch.nn.functional as F
        # IMPORTANT: Make sure this matches the model you trained (GCNConv or SAGEConv)
        from torch_geometric.nn import GCNConv 
        class GNN(torch.nn.Module):
             def __init__(self, in_channels, out_channels):
                 super(GNN, self).__init__()
                 # --- Ensure this matches your trained model ---
                 self.conv1 = GCNConv(in_channels, 64) 
                 self.conv2 = GCNConv(64, 32)
                 # ---------------------------------------------
                 self.out = torch.nn.Linear(32, out_channels)
             def forward(self, data):
                 x, edge_index = data.x, data.edge_index
                 x = self.conv1(x, edge_index)
                 x = F.relu(x)
                 x = F.dropout(x, p=0.5, training=self.training)
                 x = self.conv2(x, edge_index)
                 x = F.relu(x)
                 x = self.out(x)
                 return x

# --- Global Variables for Caching ---
# Load these heavy objects only once when the server starts
print("Loading graph, model, and data...")
try:
    G = ox.load_graphml('./data/bengaluru_graph.graphml')
    gdf_nodes, _ = ox.graph_to_gdfs(G)
    node_id_map = {osm_id: i for i, osm_id in enumerate(gdf_nodes.index)}

    model = GNN(in_channels=2, out_channels=1)
    model.load_state_dict(torch.load('gnn_model.pth', map_location=torch.device('cpu'))) # Load to CPU
    model.eval()

    graph_data = torch.load('./data/training_graph.pt', map_location=torch.device('cpu')) # Load to CPU

    # Pre-calculate predictions and costs ONCE
    print("Pre-calculating pollution predictions...")
    with torch.no_grad():
        all_predictions = model(graph_data)

    print("Assigning pollution costs to graph...")
    # --- Use the sensitive cost function ---
    sensitivity_factor = 0.10 # Or your preferred value
    for u_osm, v_osm, key, edge_data in G.edges(keys=True, data=True):
        try:
            u_idx = node_id_map[u_osm]
            v_idx = node_id_map[v_osm]
            aqi_u = all_predictions[u_idx].item()
            aqi_v = all_predictions[v_idx].item()
            avg_aqi = (aqi_u + aqi_v) / 2
            if avg_aqi < 0: avg_aqi = 0
            pollution_penalty = np.exp(avg_aqi * sensitivity_factor)
            pollution_cost = edge_data['length'] * pollution_penalty
            nx.set_edge_attributes(G, {(u_osm, v_osm, key): pollution_cost}, 'pollution_cost')
        except KeyError:
            nx.set_edge_attributes(G, {(u_osm, v_osm, key): edge_data['length']}, 'pollution_cost')

    print("Server ready.")

except FileNotFoundError:
    print("ERROR: Could not load graph or model files. Ensure they exist.")
    exit()
except Exception as e:
    print(f"ERROR during server startup: {e}")
    exit()

# --- FastAPI App ---
app = FastAPI()

# Define request body model
class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

@app.post("/get_clean_route/")
async def get_clean_route(request: RouteRequest):
    print(f"Received request: Start=({request.start_lat},{request.start_lon}), End=({request.end_lat},{request.end_lon})")
    try:
        # 1. Find nearest nodes
        orig_node = ox.nearest_nodes(G, Y=request.start_lat, X=request.start_lon)
        dest_node = ox.nearest_nodes(G, Y=request.end_lat, X=request.end_lon)

        # 2. Calculate shortest path using pre-calculated 'pollution_cost'
        print("Calculating cleanest route...")
        clean_route_nodes = nx.shortest_path(
            G, orig_node, dest_node, weight='pollution_cost'
        )
        print(f"Found clean route with {len(clean_route_nodes)} nodes.")

        # 3. Get coordinates for the route nodes
        route_coords = []
        for node_id in clean_route_nodes:
            node_data = G.nodes[node_id]
            route_coords.append((node_data['x'], node_data['y'])) # (lon, lat) order for GeoJSON

        # You might also want to return the 'fastest' route for comparison
        # fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight='length')
        # fastest_coords = [...] 

        # Return as GeoJSON LineString format (useful for Mapbox)
        return {
            "clean_route": {
                "type": "LineString",
                "coordinates": route_coords 
            }
            # "fastest_route": { ... } 
        }

    except nx.NetworkXNoPath:
        print("Error: No path found")
        raise HTTPException(status_code=404, detail="No path found between the specified points.")
    except Exception as e:
        print(f"Error during routing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Add a simple root endpoint for testing ---
@app.get("/")
async def root():
    return {"message": "AtmosAI Routing API is running."}

# --- Run the server (for local testing) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
