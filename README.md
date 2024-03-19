# Pipeline with chunking

#### 1. Prepare the raw data file `raw.csv` under the folder `/dataset/raw`

#### 2. Extract the text embedding after averaging the embedding of each chunk

```bash

python ExtractEmbedding.py --rawdata_dir imdb --output_dir ./dataset/imdb --device mps
```

#### 3. Train Classifier

```bash

python SVM.py --dataset_dir ./dataset/imdb_pooling --exp_name pooling --model_dir ./model/SVM
```

#### 4. Prediction

```bash
python SVM.py --dataset_dir ./dataset/imdb_pooling --pred --model_path ./model/SVM/pooling/SVM_TIME_STAMP.joblib
```
