import torch
import os
import argparse
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset
from tqdm import tqdm


def extract_embedding(texts, model, tokenizer, first=None, strategy='pooling', device='cpu', max_token_len=512, embedding_size=1024):
    embedding_list = []
    if strategy == 'pooling':
        print("Using strategy: avg_pooling among all chunks.")
        with torch.no_grad():
            cnt = 0
            for text in tqdm(texts):
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                num_chunks = (input_ids.shape[1]-1)//max_token_len + 1

                embedding = torch.zeros((1, embedding_size)).to(device)
                start_pos = 0
                end_pos = 0
                for i in range(num_chunks):
                    start_pos = end_pos
                    if i == num_chunks-1:
                        end_pos = input_ids.shape[1]
                    else:
                        end_pos += max_token_len
                    if (end_pos-start_pos > 512):
                        print(end_pos, start_pos)
                    # print(start_pos, end_pos)

                    output = model(
                        input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                    # embedding += output.pooler_output
                    embedding += output.last_hidden_state[:, 0, :]

                embedding /= num_chunks
                embedding_list.append(embedding.cpu())
                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    elif strategy == 'last':
        # Using the last chunk
        print('Using default strategy: keeping the last chunk.')
        with torch.no_grad():
            cnt = 0
            for text in tqdm(texts):
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                end_pos = input_ids.shape[1]
                start_pos = max(0, end_pos-max_token_len)

                output = model(
                    input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                # embedding = output.pooler_output
                embedding = output.last_hidden_state[:, 0, :]
                
                embedding_list.append(embedding.cpu())

                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    elif strategy == 'first':
        # Using the first chunk
        print('Using strategy: keeping the first chunk.')
        with torch.no_grad():
            cnt = 0
            for text in tqdm(texts):
                tokenized_dict = tokenizer(
                    text, truncation=False, return_tensors='pt')
                input_ids = tokenized_dict["input_ids"].to(device)
                mask = tokenized_dict["attention_mask"].to(device)
                start_pos = 0
                end_pos = min(input_ids.shape[1], max_token_len)

                output = model(
                    input_ids[:, start_pos:end_pos], mask[:, start_pos:end_pos])
                # embedding = output.pooler_output
                embedding = output.last_hidden_state[:, 0, :]

                embedding_list.append(embedding.cpu())

                cnt += 1
                # For small-size testing
                if cnt == first:
                    break
    else:
        raise Exception("Unsupported strategy!")

    return torch.cat(embedding_list)


def save_embeddings_to_csv(data, output_dir, name, unsupervised=False, embedding_size=1024):
    data_df = pd.DataFrame(data)
    colnames = ['e{}'.format(i) for i in range(embedding_size)]
    if not unsupervised:
        colnames.insert(0, 'label')
    data_df.columns = colnames

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    data_df.to_csv(os.path.join(output_dir, name), index=False)


def save_text_to_csv(data, output_dir, name):
    data_df = pd.DataFrame(data)
    colnames = ['text']
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
        default="last"
    )
    parser.add_argument(
        '--device', help="Accelerator device.", default='cpu'
    )
    parser.add_argument(
        '--unsupervised', help="Extract **ONLY** the embeddings of the text in unsupervised.csv.", action="store_true"
    )
    parser.add_argument(
        '--model', help='HuggingFace pretrained model name.', default='intfloat/multilingual-e5-large'
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

    model_name = parser.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(
        model_name, cache_dir=cache_dir, num_labels=2, label2id=label2id, id2label=id2label)
    config = AutoConfig.from_pretrained(
        model_name, cache_dir=cache_dir
    ).to_dict()
    max_token_len = config['max_position_embeddings'] - 2
    embedding_size = config['hidden_size']

    device = parser.device
    model.to(device)
    model.eval()
    print("Using device:", device)

    strategy = parser.strategy

    if parser.unsupervised:
        embedding = extract_embedding(
            unsupervisedData['text'], model=model, tokenizer=tokenizer, strategy=strategy,
            device=device, max_token_len=max_token_len, embedding_size=embedding_size)
        save_embeddings_to_csv(embedding, output_dir,
                               "unsupervised.csv", unsupervised=True, embedding_size=embedding_size)
        save_text_to_csv(
            unsupervisedData['text'], output_dir, "raw_unsupervised.csv")
    else:
        embedding = extract_embedding(
            rawdata['text'], model=model, tokenizer=tokenizer, strategy=strategy,
            device=device, max_token_len=max_token_len, embedding_size=embedding_size)
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
        raw_text = pd.Series(rawdata['text'])
        raw_train = raw_text[train_idx]
        raw_test = raw_text[test_idx]

        save_embeddings_to_csv(train_data, output_dir,
                               "train.csv", embedding_size=embedding_size)
        save_embeddings_to_csv(test_data, output_dir,
                               "test.csv", embedding_size=embedding_size)
        save_text_to_csv(raw_train, output_dir, "raw_train.csv")
        save_text_to_csv(raw_test, output_dir, "raw_test.csv")


if __name__ == "__main__":
    main()
