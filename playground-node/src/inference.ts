import * as tf from '@tensorflow/tfjs-node-gpu'
import { HFDataset, CharDataset, Dataset } from '@gpt/model'
import { loadModel, listModels, getModelInfo, warmupModel } from './model-loader'
import prompts from 'prompts'
import * as path from 'path'

/**
 * Interactive chat interface for testing GPT models with TensorFlow.js inference
 * Uses tf.tidy() throughout for proper memory management
 */
async function main() {
    console.log('╔═══════════════════════════════════════════════════╗')
    console.log('║   GPT Model Inference Test - TensorFlow.js       ║')
    console.log('╚═══════════════════════════════════════════════════╝\n')

    const backend = tf.getBackend()
    console.log(`🖥️  Backend: ${backend}`)
    console.log(`📊 Initial memory: ${JSON.stringify(tf.memory())}\n`)

    // Step 1: List and select a model
    const models = listModels('./models')

    if (models.length === 0) {
        console.error('❌ No models found in ./models directory')
        console.log('💡 Please train a model first using: npm run playground-node')
        process.exit(1)
    }

    console.log(`📂 Found ${models.length} model(s):\n`)

    // Display model information
    models.forEach((modelPath, idx) => {
        const info = getModelInfo(modelPath)
        if (info) {
            console.log(`${idx + 1}. ${path.basename(modelPath)}`)
            console.log(`   Date: ${info.date}`)
            console.log(`   Config: ${info.nLayer}L, ${info.nHead}H, ${info.nEmbd}E, Vocab: ${info.vocabSize}`)
            console.log()
        }
    })

    const { selectedModelIndex } = await prompts({
        type: 'select',
        name: 'selectedModelIndex',
        message: 'Select a model to load:',
        choices: models.map((m, idx) => ({
            title: `${idx + 1}. ${path.basename(m)}`,
            value: idx,
        })),
    })

    if (selectedModelIndex === undefined) {
        console.log('\n👋 Cancelled')
        process.exit(0)
    }

    const selectedModelPath = models[selectedModelIndex]

    // Step 2: Load the model
    const { model, params } = await loadModel(selectedModelPath)
    const { blockSize, vocabSize } = params

    console.log(`📖 Model config:`)
    console.log(`   - Block size: ${blockSize}`)
    console.log(`   - Vocab size: ${vocabSize}`)
    console.log(`   - Layers: ${params.nLayer}`)
    console.log(`   - Heads: ${params.nHead}`)
    console.log(`   - Embeddings: ${params.nEmbd}`)
    console.log()

    // Step 3: Initialize dataset/tokenizer
    const { datasetType } = await prompts({
        type: 'select',
        name: 'datasetType',
        message: 'Select tokenizer type:',
        choices: [
            { title: 'HuggingFace Subword (GPT-2 tokenizer)', value: 'hf' },
            { title: 'Character-level', value: 'char' },
        ],
    })

    if (!datasetType) {
        console.log('\n👋 Cancelled')
        model.dispose?.()
        process.exit(0)
    }

    let dataset: Dataset

    if (datasetType === 'hf') {
        const textSourceURL = 'https://raw.githubusercontent.com/Stoksweet/gpt-clone-tfjs/refs/heads/main/datasets/english-dictionary.txt'
        console.log(`\n⏳ Loading HuggingFace tokenizer...`)
        dataset = await HFDataset({ textSourceURL })
        console.log(`✅ Tokenizer loaded (vocab size: ${dataset.vocabSize})`)
    } else {
        const textSourceURL = 'https://raw.githubusercontent.com/Stoksweet/gpt-clone-tfjs/refs/heads/main/datasets/english-dictionary.txt'
        console.log(`\n⏳ Loading character-level tokenizer...`)
        dataset = await CharDataset({ textSourceURL })
        console.log(`✅ Tokenizer loaded (vocab size: ${dataset.vocabSize})`)
    }

    // Check if vocab sizes match
    if (dataset.vocabSize !== vocabSize) {
        console.warn(`\n⚠️  WARNING: Dataset vocab size (${dataset.vocabSize}) doesn't match model vocab size (${vocabSize})`)
        console.warn(`   This may cause errors. Make sure to use the same tokenizer that was used for training.\n`)
    }

    // Step 4: Warm up the model
    await warmupModel(model, blockSize)

    // Step 5: Configure generation parameters
    const { maxTokens, temperature, doSample, topK } = await prompts([
        {
            type: 'number',
            name: 'maxTokens',
            message: 'Max tokens to generate:',
            initial: 100,
            min: 1,
            max: 1000,
        },
        {
            type: 'number',
            name: 'temperature',
            message: 'Temperature (0.1-2.0):',
            initial: 0.8,
            min: 0.1,
            max: 2.0,
            increment: 0.1,
        },
        {
            type: 'confirm',
            name: 'doSample',
            message: 'Enable sampling (vs greedy)?',
            initial: true,
        },
        {
            type: prev => prev ? 'number' : null,
            name: 'topK',
            message: 'Top-K sampling (0 for disabled):',
            initial: 40,
            min: 0,
        },
    ])

    console.log('\n╔═══════════════════════════════════════════════════╗')
    console.log('║              Chat Interface Ready!                ║')
    console.log('║  Type your prompt and press Enter to generate    ║')
    console.log('║  Type "exit" or "quit" to end the session        ║')
    console.log('║  Type "mem" to check memory usage                ║')
    console.log('╚═══════════════════════════════════════════════════╝\n')

    // Step 6: Interactive chat loop
    let inferenceCount = 0

    while (true) {
        const { input } = await prompts({
            type: 'text',
            name: 'input',
            message: '💬 You:',
        })

        if (!input) continue

        const trimmedInput = input.trim().toLowerCase()

        // Handle special commands
        if (trimmedInput === 'exit' || trimmedInput === 'quit') {
            console.log('\n👋 Exiting...')
            break
        }

        if (trimmedInput === 'mem') {
            console.log('\n📊 Memory usage:')
            console.table(tf.memory())
            console.log()
            continue
        }

        inferenceCount++
        console.log(`\n🤖 Model (inference #${inferenceCount}):\n`)

        const inferenceStart = Date.now()

        // Track tensors for manual cleanup
        let inputTensor: tf.Tensor | null = null
        let outputTensor: tf.Tensor | null = null

        try {
            // Encode input text to tokens (can be async for some tokenizers)
            const tokensResult = dataset.encode(input)
            const tokens = tokensResult instanceof Promise ? await tokensResult : tokensResult

            // Create tensor from tokens (this needs to be cleaned up manually)
            inputTensor = tf.tensor2d([tokens], [1, tokens.length], 'int32')

            // Benchmark the generation
            const genStart = Date.now()

            // Generate response (the generate function already uses tf.tidy internally)
            outputTensor = await model.generate!({
                idx: inputTensor,
                maxNewTokens: maxTokens || 100,
                doSample: doSample ?? true,
                temperature: temperature || 0.8,
                topK: topK > 0 ? topK : undefined,
            })

            const genDuration = Date.now() - genStart

            // Convert to array and decode (can be async for some tokenizers)
            const outputTokens = (await outputTensor.array() as number[][])[0]
            const decodedTextResult = dataset.decode(outputTokens)
            const decodedText = decodedTextResult instanceof Promise ? await decodedTextResult : decodedTextResult

            console.log(decodedText)
            console.log(`\n⏱️  Generation time: ${genDuration}ms`)
            console.log(`📏 Tokens generated: ${outputTokens.length}`)
            console.log(`🎯 Tokens/sec: ${(outputTokens.length / (genDuration / 1000)).toFixed(2)}`)

            const totalDuration = Date.now() - inferenceStart
            console.log(`⏱️  Total inference time: ${totalDuration}ms\n`)

        } catch (error) {
            console.error('❌ Error during inference:', error)
            console.log()
        } finally {
            // Clean up tensors manually
            if (inputTensor) {
                inputTensor.dispose()
            }
            if (outputTensor) {
                outputTensor.dispose()
            }
        }

        // Show memory after each inference
        const memInfo = tf.memory()
        console.log(`💾 Tensors in memory: ${memInfo.numTensors}`)
        console.log()
    }

    // Cleanup
    console.log('\n🧹 Cleaning up...')

    if (dataset.dispose) {
        dataset.dispose()
    }

    if (model.dispose) {
        model.dispose()
    }

    console.log(`\n📊 Final memory state:`)
    console.table(tf.memory())

    console.log('\n✅ Session ended. Goodbye! 👋\n')
}

// Run the application
main().catch(error => {
    console.error('\n❌ Fatal error:', error)
    process.exit(1)
})
