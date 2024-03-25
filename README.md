# Pipeline with chunking

#### 1. Prepare the raw data file `raw.csv` under the folder `/dataset/raw`

#### 2. Extract the text embedding after averaging the embedding of each chunk

```bash

python ExtractEmbedding.py --rawdata_dir imdb --output_dir ./dataset/imdb  --model allenai/longformer-base-4096 --device mps --tiny_mode
```
Other models:

intfloat/multilingual-e5-large

yiyanghkust/finbert-pretrain

#### 3. Train Classifier
The `dataset_dir` here should be an output_dir created by `ExtractEmbedding.py`.
```bash

python SVM.py --dataset_dir ./dataset/imdb --exp_name SVM --model_dir ./model/
```

```bash

python MLP.py --dataset_dir ./dataset/imdb --exp_name MLP --model_dir ./model/ --num_epochs 8
```

#### 4. Prediction

```bash
python SVM.py  --pred --dataset_dir ./dataset/imdb --model_path ./model/SVM/pooling/SVM_TIME_STAMP.joblib
```

```bash
python MLP.py --pred --dataset_dir ./dataset/imdb --model_path ./model/MLP/pooling/MLP_TIME_STAMP.pt
```