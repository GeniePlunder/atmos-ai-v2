# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed by geopandas/osmnx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# Make sure gnn_model.pth and the data folder are included
COPY ./main.py .
COPY ./train_model.py . # Or 3_train_model.py (needed for GNN class definition)
COPY ./gnn_model.pth .
COPY ./data ./data

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, can be set during deployment)
# ENV NAME World

# Run main.py when the container launches using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]