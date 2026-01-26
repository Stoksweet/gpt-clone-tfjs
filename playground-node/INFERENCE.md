# Model Inference Testing

This directory contains tools for testing trained GPT models using TensorFlow.js inference.

## Files

- **`model-loader.ts`**: Utilities for loading saved models, listing available models, and warming up models
- **`inference.ts`**: Interactive CLI chat interface for testing models
- **`index.ts`**: Training script

## Usage

### 1. Train a Model (if you haven't already)

```bash
npm run start -- my-model-name
```

This will save your model to `./models/my-model-name.json`

### 2. Run the Inference Interface

```bash
npm run inference
```

This will:
1. Show all available models in the `./models` directory
2. Let you select a model to load
3. Let you choose a tokenizer (HuggingFace subword or character-level)
4. Configure generation parameters (max tokens, temperature, sampling, top-K)
5. Start an interactive chat where you can test the model

### Interactive Commands

Once in the chat interface:
- **Type any text**: Generate a response from the model
- **`mem`**: Check current TensorFlow.js memory usage
- **`exit` or `quit`**: Exit the interface

## Features

### Memory Management

The implementation uses proper memory management with:
- Manual `dispose()` calls for tensors created outside of model operations
- `tf.tidy()` is already used internally in the model's `generate()` function
- Memory monitoring available via the `mem` command

### Performance Metrics

For each inference, you'll see:
- **Generation time**: Time spent generating tokens
- **Total inference time**: Including encoding/decoding
- **Tokens generated**: Number of output tokens
- **Tokens/sec**: Generation speed

### First Inference Warmup

The tool automatically performs a warmup inference when loading a model. This initializes all TensorFlow.js operations and makes subsequent inferences faster.

## Generation Parameters

- **Max Tokens**: Maximum number of tokens to generate (1-1000)
- **Temperature**: Controls randomness (0.1-2.0)
  - Lower = more deterministic
  - Higher = more creative/random
- **Sampling**: Enable/disable sampling vs greedy decoding
- **Top-K**: Limit sampling to top K tokens (0 to disable)

## Example Session

```
╔═══════════════════════════════════════════════════╗
║   GPT Model Inference Test - TensorFlow.js       ║
╚═══════════════════════════════════════════════════╝

🖥️  Backend: tensorflow
📊 Initial memory: {"numTensors":0,"numDataBuffers":0}

📂 Found 2 model(s):

1. model-2026-01-26.json
   Date: 2026-01-26T10:30:45Z
   Config: 4L, 4H, 128E, Vocab: 50257

2. gpt-mini-test.json
   Date: 2026-01-25T15:20:10Z
   Config: 6L, 6H, 192E, Vocab: 256

? Select a model to load: › model-2026-01-26.json

📦 Loading model from ./models/model-2026-01-26.json...
🔧 Building model architecture...
⚙️  Loading weights...
✅ Model loaded successfully!

🔥 Warming up model (first inference)...
✅ Warmup complete in 1250ms

? Max tokens to generate: › 100
? Temperature (0.1-2.0): › 0.8
? Enable sampling (vs greedy)? › Yes
? Top-K sampling (0 for disabled): › 40

╔═══════════════════════════════════════════════════╗
║              Chat Interface Ready!                ║
║  Type your prompt and press Enter to generate    ║
║  Type "exit" or "quit" to end the session        ║
║  Type "mem" to check memory usage                ║
╚═══════════════════════════════════════════════════╝

💬 You: › Once upon a time

🤖 Model (inference #1):

Once upon a time in a land far away, there lived a wise old wizard...

⏱️  Generation time: 850ms
📏 Tokens generated: 100
🎯 Tokens/sec: 117.65
⏱️  Total inference time: 875ms

💾 Tensors in memory: 45

💬 You: › exit

👋 Exiting...
🧹 Cleaning up...

📊 Final memory state:
┌─────────────────┬────────┐
│ numTensors      │ 0      │
│ numDataBuffers  │ 0      │
└─────────────────┴────────┘

✅ Session ended. Goodbye! 👋
```

## Memory Management Details

The inference pipeline ensures proper cleanup:

1. **Input tensors**: Created and disposed manually after generation
2. **Output tensors**: Created by model.generate() and disposed after decoding
3. **Internal tensors**: Automatically managed by the model's internal tf.tidy() calls
4. **Dataset**: Disposed at the end of the session

## Troubleshooting

### "No models found"
Make sure you've trained at least one model first using `npm run start`

### Vocab size mismatch warning
This happens when the tokenizer vocab size doesn't match the model's trained vocab size. Make sure to use the same tokenizer type (char vs subword) that was used during training.

### Out of memory
If you encounter memory issues:
- Check memory usage with the `mem` command
- Reduce max tokens
- Restart the inference interface to clear all tensors
