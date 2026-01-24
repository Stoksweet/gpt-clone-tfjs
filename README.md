# GPT Clone

A TypeScript implementation of a GPT-style language model on TensorFlow.js. This project is optimized for learning and experimentation, supporting both character-level and subword tokenization (via transformers.js).

## Project Structure

- **`gpt/`**: The core library containing the model definition, tokenizer layers, and training logic.
  - `src/model.ts`: The definition of the GPT model architecture.
  - `src/dataset.ts`: Character-level tokenizer/dataset.
  - `src/dataset-subword.ts`: Subword-level tokenizer (Byte Pair Encoding) using `@xenova/transformers`.
  - `src/trainer.ts`: Training loop implementation.
- **`playground-node/`**: A Node.js environment for training and experimenting with the model.
  - `src/index.ts`: The main entry point script for training and generating text.

## Installation

This project is set up as an NPM workspace. To install dependencies:

```bash
npm install
```

## Usage

### Training

To run the training playground script:

```bash
# Basic run (saves to default timestamped path)
npm run playground-node

# Run with a custom model name
# This will save the model to playground-node/models/my-custom-model.json
npm run playground-node -- -- my-custom-model
```

### Configuration

You can customize training parameters in `playground-node/src/index.ts`:
- **Tokenizer**: Switch between `HFDataset` (Subword) and `CharDataset` (Character).
- **Model Size**: Adjust `CONFIG` (e.g., `gpt-pico`, `gpt-mini`).
- **Hyperparameters**: Modify `batchSize`, `maxIters`, `learningRate`, etc.

### Model Saving

The training script automatically saves the model weights and configuration to the `models/` directory upon completion. The save file is a JSON containing:
- `params`: Model configuration (layers, heads, etc.)
- `weights`: The trained parameters of the model.
- `date`: Timestamp of the save.

## Advanced

### Subword Tokenization

The project supports using Hugging Face tokenizers (default: `Xenova/gpt2`). To use a different tokenizer, pass the `tokenizerName` to `HFDataset`:

```typescript
const dataset = await HFDataset({ 
    textSourceURL: '...', 
    tokenizerName: 'Xenova/bert-base-uncased' 
});
```

## Docker Support (GPU)

To train the model using Docker with GPU support (CUDA 11.2 / cuDNN 8.1), you can use the provided Dockerfile and helper script.

### Prerequisites
- Docker
- NVIDIA GPU Driver
- NVIDIA Container Toolkit (allows `docker run --gpus all`)

### Quick Start

1. **Build and entering the container:**

   You can use the helper script:
   ```bash
   ./run-docker.sh
   ```

   Or run manually:
   ```bash
   docker build -t gpt-clone-trainer .
   docker run --gpus all -it --rm -v $(pwd):/app -w /app gpt-clone-trainer bash
   ```

2. **Inside the container:**

   If your local `node_modules` are not present or meaningful for the Linux environment, install them:
   ```bash
   npm install
   ```

   Then run the training:
   ```bash
   npm run playground-node
   ```

