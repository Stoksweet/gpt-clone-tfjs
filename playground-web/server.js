import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { GPT, HFDataset, CharDataset } from '../gpt/index.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;
const MODELS_DIR = path.join(__dirname, '../playground-node/models');

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

let loadedModel = null;
let loadedDataset = null;
let currentConfig = null;

// List all available models
app.get('/api/models', (req, res) => {
    if (!fs.existsSync(MODELS_DIR)) return res.json([]);
    const files = fs.readdirSync(MODELS_DIR)
        .filter(f => f.endsWith('.json'))
        .map(f => {
            const stats = fs.statSync(path.join(MODELS_DIR, f));
            try {
                const content = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, f), 'utf-8'));
                return {
                    id: f,
                    name: f.replace('.json', ''),
                    date: content.date || stats.mtime,
                    params: content.params
                };
            } catch (e) { return null; }
        })
        .filter(Boolean);
    res.json(files);
});

// Select and Load a model in memory on the server
app.post('/api/select-model', async (req, res) => {
    const { id, tokenizerType } = req.body;
    const filePath = path.join(MODELS_DIR, id);

    if (!fs.existsSync(filePath)) return res.status(404).send('Model not found');

    try {
        console.log(`Loading model: ${id}`);
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

        // Cleanup existing model if any
        if (loadedModel) {
            loadedModel.dispose?.();
        }
        if (loadedDataset) {
            loadedDataset.dispose?.();
        }

        // Initialize Model
        loadedModel = GPT(data.params);
        loadedModel.build();
        await loadedModel.setWeights(data.weights);
        currentConfig = data.params;

        // Initialize Dataset/Tokenizer
        const textSourceURL = 'https://raw.githubusercontent.com/Stoksweet/gpt-clone-tfjs/refs/heads/main/datasets/english-dictionary.txt';
        if (tokenizerType === 'hf') {
            loadedDataset = await HFDataset({ textSourceURL });
        } else {
            loadedDataset = await CharDataset({ textSourceURL });
        }

        res.json({ status: 'ready', params: data.params });
    } catch (err) {
        console.error(err);
        res.status(500).send(err.message);
    }
});

// Chat/Inference endpoint with streaming (SSE)
app.get('/api/chat', async (req, res) => {
    const {
        prompt,
        maxTokens = 200,
        temperature = 0.8,
        doSample = 'true',
        topK = 40
    } = req.query;

    if (!loadedModel || !loadedDataset) return res.status(400).send('Model not loaded');

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const startTime = Date.now();
    let tokenCount = 0;

    try {
        const tokensResult = loadedDataset.encode(prompt);
        const tokens = tokensResult instanceof Promise ? await tokensResult : tokensResult;

        // Context window clipping
        const blockSize = currentConfig.blockSize;
        const clippedTokens = tokens.slice(-blockSize);

        const inputTensor = tf.tensor2d([clippedTokens], [1, clippedTokens.length], 'int32');

        const generationStartTime = Date.now();

        // Generation loop leveraging model.generate logic but streaming tokens
        await loadedModel.generate({
            idx: inputTensor,
            maxNewTokens: parseInt(maxTokens),
            doSample: doSample === 'true',
            temperature: parseFloat(temperature),
            topK: parseInt(topK) > 0 ? parseInt(topK) : undefined
        }, async (token) => {
            tokenCount++;
            const charResult = loadedDataset.decode([token]);
            const char = charResult instanceof Promise ? await charResult : charResult;
            res.write(`data: ${JSON.stringify({ token: char })}\n\n`);
        });

        const endTime = Date.now();
        const generationTime = endTime - generationStartTime;
        const totalTime = endTime - startTime;

        const stats = {
            generationTime,
            totalTime,
            tokenCount,
            tokensPerSec: (tokenCount / (generationTime / 1000)).toFixed(2),
            memory: tf.memory().numTensors
        };

        res.write(`data: ${JSON.stringify({ stats })}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
        inputTensor.dispose();
    } catch (err) {
        console.error(err);
        res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
        res.end();
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Premium GPT Server running at http://localhost:${PORT}`);
});
