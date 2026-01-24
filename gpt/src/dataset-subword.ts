import * as tf from '@tensorflow/tfjs';
import { AutoTokenizer } from '@xenova/transformers';
import { Dataset, DatasetGetBatchParams, DatasetParams } from './types';

export interface HFDatasetParams extends DatasetParams {
    tokenizerName?: string;
}

/**
 * Creates a subword-level dataset using Hugging Face's transformers.js.
 */
export async function HFDataset(args: HFDatasetParams): Promise<Dataset> {
    const { textSourceURL, textSource = '', tokenizerName = 'Xenova/gpt2' } = args;

    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerName);
    const text: string = textSourceURL ? await (await fetch(textSourceURL)).text() : textSource;

    // Tokenize the entire text
    // encode returns number[] directly (or Promise<number[]>)
    const dataArray = (await tokenizer.encode(text)) as unknown as number[];
    // verify it is array of numbers

    const vocabSize = tokenizer.model.vocab.length;
    const dataSize = dataArray.length;

    // Data encoders/decoders
    const encode = async (s: string) => {
        const ids = await tokenizer.encode(s);
        return ids as unknown as number[];
    };


    const decode = (a: number[]) => {
        return tokenizer.decode(a);
    };

    // train and test splits
    const data = tf.tensor(dataArray, [dataSize], 'int32');
    const n = Math.floor(0.9 * dataSize);
    const trainData: tf.Tensor = data.slice(0, n);
    const valData: tf.Tensor = data.slice(n);

    const getBatch = (args: DatasetGetBatchParams) =>
        tf.tidy(() => {
            const { split, blockSize, batchSize } = args;

            const currentData = split === 'train' ? trainData : valData;
            const currentDataSize = currentData.shape[0];

            // Randomly sample indices
            const maxval = currentDataSize - blockSize - 1;
            const ix = tf.randomUniform([batchSize], 0, maxval, 'int32'); // (B)
            const ranges = tf.range(0, blockSize, 1, 'int32').expandDims(0); // (1,T)
            const indices = ix.expandDims(1).add(ranges); // (B,T)

            const x = tf.gather(currentData, indices); // (B,T)
            const y = tf.gather(currentData, indices.add(tf.scalar(1, 'int32'))); // (B,T)

            return { x, y };
        });

    const dispose = () => {
        data.dispose();
        trainData.dispose();
        valData.dispose();
    };

    return {
        textSourceURL,
        vocabSize,
        vocabulary: Object.keys(tokenizer.model.vocab),
        dataSize,
        text,
        getBatch,
        encode,
        decode,
        dispose
    };
}
