import * as tf from '@tensorflow/tfjs-node-gpu'
import { GPT, Model } from '@gpt/model'
import * as fs from 'fs'
import * as path from 'path'

export interface LoadedModel {
    model: Model
    params: any
    date: string
}

/**
 * Load a saved model from a JSON file with proper memory management
 */
export async function loadModel(modelPath: string): Promise<LoadedModel> {
    console.log(`\n📦 Loading model from ${modelPath}...`)

    if (!fs.existsSync(modelPath)) {
        throw new Error(`Model file not found: ${modelPath}`)
    }

    const data = JSON.parse(fs.readFileSync(modelPath, 'utf-8'))

    if (!data.params || !data.weights) {
        throw new Error(`Invalid model file: missing params or weights`)
    }

    // Reconstruct model with saved params
    const model = GPT(data.params)

    console.log('🔧 Building model architecture...')
    model.build()

    // Load weights into the model
    console.log('⚙️  Loading weights...')
    await model.setWeights!(data.weights)

    console.log('✅ Model loaded successfully!\n')

    return {
        model,
        params: data.params,
        date: data.date || 'unknown',
    }
}

/**
 * List all available model files in the models directory
 */
export function listModels(modelsDir: string = './models'): string[] {
    if (!fs.existsSync(modelsDir)) {
        console.warn(`Models directory not found: ${modelsDir}`)
        return []
    }

    const files = fs.readdirSync(modelsDir)
        .filter(f => f.endsWith('.json'))
        .map(f => path.join(modelsDir, f))
        .sort((a, b) => {
            // Sort by modification time, newest first
            const statA = fs.statSync(a)
            const statB = fs.statSync(b)
            return statB.mtime.getTime() - statA.mtime.getTime()
        })

    return files
}

/**
 * Get model metadata without loading the full model
 */
export function getModelInfo(modelPath: string): any {
    if (!fs.existsSync(modelPath)) {
        return null
    }

    try {
        const data = JSON.parse(fs.readFileSync(modelPath, 'utf-8'))
        return {
            path: modelPath,
            date: data.date || 'unknown',
            params: data.params,
            vocabSize: data.params?.vocabSize,
            blockSize: data.params?.blockSize,
            nLayer: data.params?.nLayer,
            nHead: data.params?.nHead,
            nEmbd: data.params?.nEmbd,
        }
    } catch (error) {
        return null
    }
}

/**
 * Warm up the model with a dummy inference to initialize all tensor operations
 * This ensures the first real inference is faster
 */
export async function warmupModel(model: Model, blockSize: number): Promise<number> {
    console.log('🔥 Warming up model (first inference)...')

    const startTime = Date.now()

    // Create dummy input
    const dummyInput = tf.ones([1, blockSize], 'int32')

    try {
        const output = await model.generate!({
            idx: dummyInput,
            maxNewTokens: 5,
            doSample: false,
        })

        // Clean up output tensor
        output.dispose()
    } finally {
        // Clean up input tensor
        dummyInput.dispose()
    }

    const duration = Date.now() - startTime
    console.log(`✅ Warmup complete in ${duration}ms\n`)

    return duration
}
