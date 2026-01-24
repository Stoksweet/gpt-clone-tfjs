import * as tf from '@tensorflow/tfjs-node-gpu'
import { CONFIG, GPT, HFDataset, CharDataset, Trainer } from '@gpt/model'

async function start() {
  const backend = tf.getBackend()
  console.log(`Current backend: ${backend}`)

  const textSourceURL = 'https://raw.githubusercontent.com/trekhleb/homemade-gpt-js/refs/heads/main/playground-web/public/dataset-tinyshakespeare.txt'
  const dataset = await HFDataset({ textSourceURL })

  const batchSize = 16
  const blockSize = 16
  const maxIters = 2800
  const evalInterval = 200
  const evalIterations = 10
  const learningRate = 7e-4

  const model = GPT({
    ...CONFIG['gpt-mini'],
    blockSize,
    vocabSize: dataset.vocabSize,
  })
  console.log('\nModel summary:', model.summary())

  // Parse command line arguments for model name
  const args = process.argv.slice(2);
  const modelNameInput = args[0];
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const modelName = modelNameInput || `model-${timestamp}`;
  const modelPath = `file://./models/${modelName}`;

  console.log(`\nModel will be saved to: ${modelPath}`);

  console.log('\nStart training:')
  const trainer = Trainer({
    model,
    dataset,
    params: {
      learningRate,
      evalInterval,
      evalIterations,
      maxIters,
      batchSize,
      blockSize,
    },
    callbacks: {
      onEval: (params) => {
        console.log(params)
      },
    },
  })
  await trainer.train()

  console.log('\nStart generation:')
  const generated = await model.generate({
    idx: tf.ones([1, 1], 'int32'),
    maxNewTokens: 500,
    doSample: true,
    topK: undefined,
  })
  console.log(dataset.decode(((await generated.array()) as number[][])[0]))

  console.log(`\nSaving model to ${modelPath}...`);

  const fs = require('fs');
  const path = require('path');

  // Ensure directory exists
  const dir = path.dirname(modelPath.replace('file://', ''));
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const weights = await model.getWeights!();
  const artifact = {
    date: new Date().toISOString(),
    params: model.params,
    weights
  };

  const filePath = modelPath.replace('file://', '') + '.json';
  fs.writeFileSync(filePath, JSON.stringify(artifact, null, 2));
  console.log(`Model saved to ${filePath}`);

  console.log('\nDisposing the model and dataset')
  dataset.dispose()
  generated.dispose()
  model?.dispose?.()

  console.log('Model Summary: ', model.summary());

  console.log('\nDebug memory consumption:')
  console.table(tf.memory())
}

start()
