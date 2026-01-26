# Quick Start Guide - Model Inference

## What We Built

A complete TensorFlow.js model inference testing system with:

✅ **Model Loader** - Load saved models from disk  
✅ **Interactive Chat CLI** - Test models with prompts  
✅ **Memory Management** - Proper tensor cleanup with manual dispose  
✅ **Performance Metrics** - Track inference speed and memory  
✅ **Multi-Tokenizer Support** - Works with char-level and subword tokenizers  

## File Structure

```
playground-node/
├── src/
│   ├── index.ts          # Training script (existing)
│   ├── model-loader.ts   # 🆕 Load and manage saved models
│   └── inference.ts      # 🆕 Interactive chat interface
├── models/               # Saved models (created by training)
│   └── *.json           # Model files
├── INFERENCE.md         # 🆕 User documentation
├── IMPLEMENTATION.md    # 🆕 Technical documentation
└── package.json         # Updated with "inference" script
```

## How to Use

### Step 1: Train a Model (if needed)

```bash
cd playground-node
npm run start -- my-model-name
```

### Step 2: Run Inference

```bash
npm run inference
```

### Step 3: Select Model & Configure

The interface will guide you through:
1. Selecting a trained model
2. Choosing a tokenizer
3. Configuring generation parameters
4. Testing with prompts

## Memory Management Explained

### The Challenge
- `tf.tidy()` doesn't work with async functions
- Model operations (`generate()`, `encode()`, `decode()`) are async
- Need manual tensor cleanup

### The Solution
```typescript
// Create tensors
const inputTensor = tf.tensor2d([tokens], [1, tokens.length], 'int32')

try {
  // Use tensors
  const output = await model.generate!({ idx: inputTensor, ... })
  // ... process output
} finally {
  // Always cleanup
  inputTensor.dispose()
  output.dispose()
}
```

## Key Features

### 1. Model Selection
```
📂 Found 2 model(s):

1. model-2026-01-26.json
   Date: 2026-01-26T10:30:45Z
   Config: 4L, 4H, 128E, Vocab: 50257
```

### 2. Warmup
Performs first inference to initialize TF.js operations:
```
🔥 Warming up model (first inference)...
✅ Warmup complete in 1250ms
```

### 3. Performance Metrics
```
⏱️  Generation time: 850ms
📏 Tokens generated: 100
🎯 Tokens/sec: 117.65
⏱️  Total inference time: 875ms

💾 Tensors in memory: 45
```

### 4. Memory Monitoring
Type `mem` during chat:
```
📊 Memory usage:
┌─────────────────┬────────┐
│ numTensors      │ 45     │
│ numDataBuffers  │ 45     │
└─────────────────┴────────┘
```

## Dependencies Added

```json
{
  "dependencies": {
    "prompts": "^2.4.2",
    "@types/prompts": "^2.4.9"
  }
}
```

## Testing Memory Management

Run inference multiple times and check that tensor count stays constant:

```bash
💬 You: > prompt 1
💾 Tensors in memory: 45

💬 You: > prompt 2
💾 Tensors in memory: 45  # ✅ Same count = no leak!

💬 You: > prompt 3
💾 Tensors in memory: 45  # ✅ Still stable!
```

## Next Steps

You can now:
1. ✅ Train models with `npm run start`
2. ✅ Test models with `npm run inference`
3. ✅ Monitor memory usage
4. ✅ Benchmark performance
5. ✅ Experiment with different parameters

## Troubleshooting

**No models found?**
→ Train one first: `npm run start`

**Vocab size mismatch?**
→ Use the same tokenizer type as training

**Memory growing?**
→ Check the `mem` command - should stay stable

**Slow first inference?**
→ Normal! Warmup initializes TF.js operations

## Documentation

- **INFERENCE.md** - Complete user guide
- **IMPLEMENTATION.md** - Technical details on memory management
- **README.md** - Updated with inference section

Enjoy testing your models! 🚀
