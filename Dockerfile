# Base image with CUDA 11.2 and cuDNN 8.1
FROM tensorflow/tensorflow:2.11.0-gpu

# Install Node.js 20
# TensorFlow 2.11.0 image is based on Ubuntu 20.04
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Note: We rely on the user mounting the code at runtime for development
# but strictly speaking, a Dockerfile should describe how to build the app.
# We will copy the package definitions to allow 'docker build' to cache dependencies
# if the user chooses to build a standalone image.

COPY package.json package-lock.json ./
COPY gpt/package.json ./gpt/
COPY playground-node/package.json ./playground-node/

# Install dependencies (this will build native bindings for the container environment)
# If you mount your local directory, you might need to run 'npm install' again inside the container.
RUN npm ci

# Copy the rest of the source code
COPY . .

# Default command
CMD ["bash"]
