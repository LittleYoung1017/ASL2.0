ASL is a common diagram for audio splicing detection and localization.
it is a LCNN+LSTM model.


## Data preprocessing
1. 对两个audio数据集分别划分训练集和验证集，将音频数据分割成固定长度音频，两两随机拼接，设置分割长度为3s。
```

python utils.py
    --type dividing_resample_spliting_concat \
    --drop_last 1 \
    --s_path /home/yangruixiong/dataset/new_splicing_detection_dataset/guangda-bank-3s-8k/train \
    --s_path2 /home/yangruixiong/dataset/new_splicing_detection_dataset/southern-power-grid-3s-8k/train \
    --t_path /home/yangruixiong/dataset/new_splicing_detection_dataset/concat-gdbank-spg-3s-8k/train \
    --s_sr 8000 \
    --s_sr2 8000 \
    --t_sr 8000 \
    --cutting_time 3
```
2. 对分割和拼接后的数据提取特征并保存为npz文件。
```
python downsampling_mfcc_lfcc_concat2.py
```

## Train
3. 少量数据进行训练：
```
python trainer.py
```
4. 大数据集分批进行训练
```
python trainer2.py
```

## Inference
5. 
```
python test.py
```

