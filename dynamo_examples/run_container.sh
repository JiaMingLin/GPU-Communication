#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
#docker build -t dynamo-runner .

# Run the container with GPU support
echo "Running container with GPU support..."
docker run -it --gpus all -v $(pwd)/app:/app dynamo-runner
