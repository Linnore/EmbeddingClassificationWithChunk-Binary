import torch
import os
import argparse
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm

MAX_TOKEN_LEN = 512
EMBEDDING_SIZE = 1024


def extract_embedding(texts, model, tokenizer, first=None, strategy='pooling', device='cpu'):
    embedding_list = []
    if strategy == 'pooling':
        with torch.no_grad():
            cnt = 0
            for text in tqdm(texts):
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                num_chunks = (input_ids.shape[1]-1)//MAX_TOKEN_LEN + 1

                embedding = torch.zeros((1, EMBEDDING_SIZE)).to(device)
                start_pos = 0
                end_pos = 0
                for i in range(num_chunks):
                    start_pos = end_pos
                    if i == num_chunks-1:
                        end_pos = input_ids.shape[1]
                    else:
                        end_pos += MAX_TOKEN_LEN
                    if (end_pos-start_pos > 512):
                        print(end_pos, start_pos)
                    # print(start_pos, end_pos)

                    output = model(
                        input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                    embedding += output.pooler_output
                embedding /= num_chunks
                embedding_list.append(embedding.cpu())
                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    elif strategy == 'last':
        # Using the last chunk
        with torch.no_grad():
            cnt = 0
            for text in tqdm(texts):
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                end_pos = input_ids.shape[1]
                start_pos = max(0, end_pos-MAX_TOKEN_LEN)

                output = model(
                    input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                embedding = output.pooler_output

                embedding_list.append(embedding.cpu())

                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    elif strategy == 'first':
        # Using the last chunk

        with torch.no_grad():
            cnt = 0
            for text in tqdm(texts):
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                start_pos = 0
                end_pos = min(input_ids.shape[1], MAX_TOKEN_LEN)

                output = model(
                    input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                embedding = output.pooler_output

                embedding_list.append(embedding.cpu())

                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    else:
        raise Exception("Unsupported strategy!")

    return torch.cat(embedding_list)


def save_to_csv(data, output_dir, name, unsupervised=False):
    data_df = pd.DataFrame(data)
    colnames = ['e{}'.format(i) for i in range(EMBEDDING_SIZE)]
    if not unsupervised:
        colnames.insert(0, 'label')
    data_df.columns = colnames

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    data_df.to_csv(os.path.join(output_dir, name), index=False)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Use language model to extract text embedding.")
    parser.add_argument(
        '--output_dir', help='Directory to save the embedding files.', default="./dataset/processed"
    )
    parser.add_argument(
        '--name', help="Output name of the data files.", default=None
    )
    parser.add_argument(
        '--rawdata_dir', help="Directory for the input raw dataset.", default="./dataset/raw"
    )
    parser.add_argument(
        '--tiny_mode', help='Tiny mode for sanity check.', action='store_true'
    )
    parser.add_argument(
        '--strategy', help="Strategy to of chunking. 'pooling': Construct the embedding by averaging among all chunks; 'first': Only use the first chunk; 'Last': Only use the last chunk.",
        default="pooling"
    )
    parser.add_argument(
        '--device', help="Accelerator device.", default='cpu'
    )
    parser.add_argument(
        '--unsupervised', help="Extract **ONLY** the embeddings of the text in unsupervised.csv.", action="store_true"
    )

    parser = parser.parse_args()

    rawdata_dir = parser.rawdata_dir
    cache_dir = "../cache/"

    if rawdata_dir == 'imdb':
        rawdata = load_dataset(
            rawdata_dir, cache_dir=cache_dir, split="train+test")
        unsupervisedData = load_dataset(
            rawdata_dir, cache_dir=cache_dir, split="unsupervised")
        
        print(rawdata, unsupervisedData)
    else:
        try:
            rawdata_dir = parser.rawdata_dir
            raw_path = os.path.join(rawdata_dir, 'raw.csv')
            rawdata = load_dataset(
                rawdata_dir, data_files={'raw': 'raw.csv', 'unsupervised': 'unsupervised.csv'})
            unsupervisedData = rawdata['unsupervised.csv']
            rawdata = rawdata['raw']
        except:
            raise Exception("Fail to load " + raw_path + "!")

    output_dir = parser.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tiny_mode = parser.tiny_mode
    if tiny_mode:
        print("Tiny mode enable.")
        rawdata = rawdata.shuffle(seed=118010142).select(
            range(int(0.001*len(rawdata))))
        unsupervisedData = unsupervisedData.shuffle(seed=118010142).select(
            range(int(0.001*len(unsupervisedData))))

    id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
    label2id = {'NEGATIVE': 0, 'POSITIVE': 1}

    tokenizer = AutoTokenizer.from_pretrained(
        'intfloat/multilingual-e5-large', cache_dir=cache_dir)
    model = AutoModel.from_pretrained(
        'intfloat/multilingual-e5-large', cache_dir=cache_dir, num_labels=2, label2id=label2id, id2label=id2label)

    device = parser.device
    model.to(device)
    print("Using device:", device)

    strategy = parser.strategy
    
    if parser.unsupervised:
        embedding = extract_embedding(
            unsupervisedData['text'], model=model, tokenizer=tokenizer, strategy=strategy, device=device
        )
        save_to_csv(embedding, output_dir, "unsupervised.csv", unsupervised=True)
    else:
        embedding = extract_embedding(
            rawdata['text'], model=model, tokenizer=tokenizer, strategy=strategy, device=device)
        label = rawdata['label']
        data = torch.cat((torch.tensor(label).unsqueeze(1), embedding), 1)

        test_size = 0.2
        np.random.seed(118010142)
        total_num = data.shape[0]
        rand_idx = np.arange(total_num)
        np.random.shuffle(rand_idx)
        test_idx = rand_idx[:int(total_num*test_size)]
        train_idx = rand_idx[int(total_num*test_size):]

        train_data = data[train_idx, :]
        test_data = data[test_idx, :]

        save_to_csv(train_data, output_dir, "train.csv")
        save_to_csv(test_data, output_dir, "test.csv")


if __name__ == "__main__":
    main()
