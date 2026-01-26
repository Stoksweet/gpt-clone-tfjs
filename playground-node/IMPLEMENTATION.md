# TensorFlow.js Inference Implementation - Technical Overview

## Architecture

The inference testing system consists of two main components with proper memory management using `tf.tidy()` where applicable:

### 1. Model Loader (`model-loader.ts`)

**Purpose**: Load saved models and prepare them for inference

**Key Functions**:

- **`loadModel(modelPath)`**: Loads a saved model from JSON
  - Reads model params and weights from disk
  - Reconstructs the GPT model architecture
  - Uses `model.setWeights()` to restore trained parameters
  - Returns model, params, and metadata

- **`listModels(modelsDir)`**: Lists all `.json` model files
  - Scans the models directory
  - Sorts by modification time (newest first)
  - Returns array of file paths

- **`getModelInfo(modelPath)`**: Reads model metadata without loading
  - Efficient way to display model information
  - No TensorFlow operations, just JSON parsing

- **`warmupModel(model, blockSize)`**: Performs first inference
  - Creates a dummy input tensor
  - Runs a short generation (5 tokens)
  - **Memory Management**: Manually disposes input and output tensors
  - Returns warmup duration for benchmarking
  - Note: Cannot use `tf.tidy()` here because `model.generate()` is async

### 2. Inference Interface (`inference.ts`)

**Purpose**: Interactive CLI for testing models

**Flow**:

1. **Model Selection**
   - Lists available models with metadata
   - User selects a model via interactive prompt

2. **Model Loading**
   - Calls `loadModel()` to load selected model
   - Displays model configuration

3. **Tokenizer Selection**
   - User chooses between HuggingFace (subword) or character-level tokenizer
   - Initializes the dataset/tokenizer
   - Validates vocab size matches model

4. **Warmup**
   - Calls `warmupModel()` to perform first inference
   - Displays warmup time

5. **Parameter Configuration**
   - User configures: maxTokens, temperature, doSample, topK
   - These parameters control generation behavior

6. **Interactive Chat Loop**
   - User enters prompts
   - System performs inference and displays results
   - Shows performance metrics
   - Supports special commands (`mem`, `exit`)

## Memory Management Strategy

### Why We Can't Use `tf.tidy()` Everywhere

`tf.tidy()` in TensorFlow.js is designed to:
- Automatically dispose of tensors created within a synchronous function
- Return a single tensor (or tensor container) that should be kept

**Problem**: `tf.tidy()` does NOT support async functions!

```typescript
// ❌ This doesn't work - tf.tidy() can't handle async functions
await tf.tidy(async () => {
  const tensor = await someAsyncOperation()
  return tensor
})
```

Since `model.generate()`, `tensor.array()`, `dataset.encode()`, and `dataset.decode()` are all async, we can't wrap the inference pipeline in `tf.tidy()`.

### Our Solution: Manual Memory Management

We use a try-finally pattern to ensure tensors are always disposed:

```typescript
let inputTensor: tf.Tensor | null = null
let outputTensor: tf.Tensor | null = null

try {
  // Create and use tensors
  inputTensor = tf.tensor2d([tokens], [1, tokens.length], 'int32')
  outputTensor = await model.generate!({ idx: inputTensor, ... })
  
  // Use the tensors...
} finally {
  // Always clean up, even if there's an error
  if (inputTensor) inputTensor.dispose()
  if (outputTensor) outputTensor.dispose()
}
```

### Where `tf.tidy()` IS Used

1. **Inside the GPT model** (`model.ts`):
   - The `apply()` function uses `tf.tidy()` for forward pass
   - The `loss()` function uses `tf.tidy()` for loss computation
   - Internal operations in the `generate()` loop use `tf.tidy()`

2. **Model building and summary**:
   - `model.build()` uses `tf.tidy()`
   - `model.summary()` uses `tf.tidy()`

These are synchronous operations, so `tf.tidy()` works perfectly.

## Inference Pipeline with Memory Management

Here's the complete flow with memory annotations:

```typescript
// 1. Encode input (no tensors created)
const tokens = await dataset.encode(input)

// 2. Create input tensor (NEEDS CLEANUP)
const inputTensor = tf.tensor2d([tokens], [1, tokens.length], 'int32')

// 3. Generate (creates many intermediate tensors, but model.generate() 
//    uses tf.tidy() internally for intermediate operations)
//    Returns ONE tensor that we need to manage
const outputTensor = await model.generate!({
  idx: inputTensor,
  maxNewTokens: 100,
  ...
})

// 4. Convert to array (no new tensors)
const outputTokens = await outputTensor.array()

// 5. Decode (no tensors)
const text = await dataset.decode(outputTokens)

// 6. CLEANUP - dispose tensors we created
inputTensor.dispose()   // Clean up input
outputTensor.dispose()  // Clean up output
```

## Type Safety Considerations

### Handling Union Types for encode/decode

The `Dataset` type defines:
```typescript
encode: (s: string) => number[] | Promise<number[]>
decode: (a: number[]) => string | Promise<string>
```

This is because:
- Character-level tokenizer: synchronous (returns `number[]` / `string`)
- HuggingFace tokenizer: asynchronous (returns `Promise<number[]>` / `Promise<string>`)

**Our solution**:
```typescript
const tokensResult = dataset.encode(input)
const tokens = tokensResult instanceof Promise 
  ? await tokensResult 
  : tokensResult
```

This handles both cases gracefully.

## Performance Monitoring

The interface tracks and displays:

1. **Generation time**: Time for `model.generate()` call
2. **Total inference time**: Including encode/decode overhead
3. **Tokens generated**: Output length
4. **Tokens/sec**: Throughput metric
5. **Memory usage**: Number of tensors in memory

## Best Practices Applied

1. ✅ **Manual disposal** for tensors created outside of model operations
2. ✅ **try-finally** blocks to ensure cleanup even on errors
3. ✅ **Memory monitoring** via `mem` command and post-inference stats
4. ✅ **Warmup inference** to initialize TF.js operations
5. ✅ **Proper async/await** handling throughout
6. ✅ **Type-safe** handling of union return types
7. ✅ **Model's internal tf.tidy()** for synchronous operations

## Testing the Implementation

To verify memory management is working:

```bash
# Run inference
npm run inference

# In the chat:
💬 You: > test prompt 1
🤖 Model: [output]
💾 Tensors in memory: 45

💬 You: > test prompt 2
🤖 Model: [output]
💾 Tensors in memory: 45  # Should be same!

💬 You: > mem
📊 Memory usage:
┌─────────────────┬────────┐
│ numTensors      │ 45     │  # Stable number = good!
│ numDataBuffers  │ 45     │
└─────────────────┴────────┘
```

If tensors keep increasing after each inference, there's a memory leak. With proper cleanup, the count should stay constant (only model weights remain in memory).

## Extending the System

To add new features:

1. **Web UI**: Replace `prompts` with Express + Socket.io
2. **Batch inference**: Process multiple prompts in parallel
3. **Streaming**: Use `onGenerateChar` callback in `model.generate()`
4. **Model comparison**: Load multiple models and compare outputs
5. **Quantization**: Add support for quantized models

All extensions should follow the same memory management patterns established here.
