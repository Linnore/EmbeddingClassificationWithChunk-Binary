# Pipeline with chunking

#### 1. Prepare the raw data file `raw.csv` under the folder `/dataset/raw`

#### 2. Extract the text embedding after averaging the embedding of each chunk

```bash

python ExtractEmbedding.py --rawdata_dir imdb --output_dir ./dataset/imdb --device mps --model allenai/longformer-base-4096 --tiny_mode
```

#### 3. Train Classifier

```bash

python SVM.py --dataset_dir ./dataset/imdb_pooling --exp_name pooling --model_dir ./model/SVM
```

```bash

python MLP.py --dataset_dir ./dataset/imdb_pooling --exp_name pooling --model_dir ./model/MLP --num_epochs 8
```

#### 4. Prediction

```bash
python SVM.py  --pred --dataset_dir ./dataset/imdb_pooling --model_path ./model/SVM/pooling/SVM_TIME_STAMP.joblib
```

```bash
python MLP.py --pred --dataset_dir ./dataset/imdb_pooling --model_path ./model/MLP/pooling/MLP_TIME_STAMP.pt
```