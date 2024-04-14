# Binary Classification using BERT-based Model with Chunking Option.
#### 0. Requirement:
```bash
pip install torch, torchvision, scikit-learn, transformers, accelerate, datasets, evaluate, joblib, scipy, tensorboard, tqdm
```
#### 1. Prepare the raw data file `raw.csv` under the folder `/dataset/raw` 
Skip this step for example run on IMDB.

The raw data file should contain only two columns: label, text. The label data field should be a string, where labels are joined via ",". The text columns should also be a string.


#### 2. Extract the text embedding after averaging the embedding of each chunk

```bash

python ExtractEmbedding.py --rawdata_dir imdb --output_dir ./dataset/imdb_mini  --model intfloat/multilingual-e5-large --device mps --strategy pooling --tiny_mode
```

```bash

python ExtractEmbedding.py --unsupervised --rawdata_dir imdb --output_dir ./dataset/imdb_mini  --model intfloat/multilingual-e5-large --device mps --strategy pooling --tiny_mode 
```

Note:
 - Please specify appropriate device for acceleration, e.g., cuda, cpu (if no accelerator available).
 - Strategy "first" or "last" implements the same embedding extraction without chunking.
 - You can change `imdb` to your data folder prepared in step 1.
 - You can change to other BERT-based model available in Huggingface Hub or provide a local pretrained model directory as the --model.
   - allenai/longformer-large-4096
   - intfloat/multilingual-e5-large
   - yiyanghkust/finbert-pretrain
   - efederici/e5-base-multilingual-4096
  

#### 3. Train Classifier
The `dataset_dir` here should be an output_dir created by `ExtractEmbedding.py`.
```bash

python SVM.py --dataset_dir ./dataset/imdb_mini --exp_name SVM --model_dir ./model/
```

```bash

python MLP.py --dataset_dir ./dataset/imdb_mini --exp_name MLP --model_dir ./model/ --num_epochs 8
```

Note:
 - The dataset_dir should be a output_dir from step 2.
  
#### 4. Prediction
Prediction requires an `unsupervised.csv` input file under dataset_dir.

```bash
python SVM.py  --pred --dataset_dir ./dataset/imdb_mini --model_path ./model/SVM/SVM_TIME_STAMP.joblib
```

```bash
python MLP.py --pred --dataset_dir ./dataset/imdb_mini --model_path ./model/MLP/MLP_TIME_STAMP.pt
```

