#!/bin/bash

# Image name
IMAGE_NAME="gpt-clone-trainer"

echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

echo "Starting container..."
echo "Tips: Inside the container, run 'npm install' if your local node_modules are missing or incompatible."
echo "Then run: npm run playground-node"

# Run the container with GPU support
# We map the current directory to /app so changes persist
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -w /app \
  $IMAGE_NAME \
  bash
