#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t matrix-runner .

# Run the container with the newly built image
echo "Running container..."
docker run --rm -v $(pwd)/app:/app matrix-runner
