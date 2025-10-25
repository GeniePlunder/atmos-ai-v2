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
RUN pip install --no-cache-dir --upgrade pip
# Add a meaningless argument to potentially break cache
RUN pip install --no-cache-dir -r requirements.txt --cache-dir /tmp/pip-cache-bust

# --- Debugging Step ---
# List contents of /app before copying application code
RUN echo "--- Contents of /app before copying code: ---" && ls -la /app && echo "--------------------------------------------"
# --- End Debugging Step ---

# Copy the rest of your application code into the container
COPY ./main.py .
COPY ./train_model.py . # This is the line that was failing
COPY ./gnn_model.pth .
COPY ./data ./data

# Make port 8000 available
EXPOSE 8000

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]