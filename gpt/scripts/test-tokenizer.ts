import { HFDataset } from '../src/dataset-subword';
import * as tf from '@tensorflow/tfjs';

async function test() {
    console.log('Testing HFDataset with GPT-2 tokenizer...');

    const text = 'Hello, this is a test of the subword tokenizer integration. It should be more efficient than character-level tokenization.';

    const dataset = await HFDataset({
        textSource: text,
        tokenizerName: 'gpt2'
    });

    console.log('Vocab Size:', dataset.vocabSize);
    console.log('Data Size (tokens):', dataset.dataSize);
    console.log('Original Text Length (chars):', text.length);

    const encoded = await dataset.encode('Beginning of document.') as number[];
    console.log('Encoded "Beginning of document.":', encoded);

    const decoded = dataset.decode(encoded);
    console.log('Decoded back:', decoded);

    if (decoded === 'Beginning of document.') {
        console.log('✅ Encode/Decode test passed!');
    } else {
        console.log('❌ Encode/Decode test failed!');
    }

    const { x, y } = dataset.getBatch({
        split: 'train',
        blockSize: 8,
        batchSize: 4
    });

    console.log('Batch X shape:', x.shape);
    console.log('Batch Y shape:', y.shape);

    if (x.shape[0] === 4 && x.shape[1] === 8) {
        console.log('✅ Batch generation shape test passed!');
    } else {
        console.log('❌ Batch generation shape test failed!');
    }

    dataset.dispose();
}

test().catch(console.error);
