ASL is a common diagram for audio splicing detection and localization and use a LCNN+LSTM model.


## Data preprocessing
1. 在config中修改预处理数据保存位置：
```
    "data":{
        "downsample_data":"path_to_save_npz_file"
    }
```
2. 对两个audio数据集分别划分训练集和验证集，将音频数据分割成固定长度音频，两两随机拼接，设置分割长度为3s。再对切割好的音频和拼接生成的音频提取MFCC+LFCC特征，两个特征合并在一起，最终保存为npz文件。
```
    python utils.py
        --type audio_preprocessing \
        --drop_last 1 \
        --s_path path_to_dataset1 \
        --s_path2 path_to_dataset2 \
        --t_path path_to_save_produced_dataset \
        --s_sr 8000 \  
        --s_sr2 8000 \
        --t_sr 8000 \
        --cutting_time 3
```
生成的目录如下：
```
    -dataset1
      -train
        -original_1.wav #3s split audio file
        -original_2.wav
         ...
      -val
        -original_1.wav
        -original_2.wav
         ... 
    -dataset2
      -train
      -val
    -concat_dataset1_dataset2
      -train
      -val

    -dataset1_mfcc_lfcc
      -train_data_mfcc_lfcc0.npz
      -train_data_mfcc_lfcc1.npz
       ...
      -test_data_mfcc_lfcc0.npz
       ...
    -dataset2_mfcc_lfcc
    -concat_dataset1_dataset2_mfcc_lfcc

```

单独对数据提取保存MFCC+LFCC特征时使用：
```
    python downsampling_mfcc_lfcc_concat2.py
```

## Train
3. 进行少数据量训练：
```
    python trainer.py
```
大数据集训练执行：
```
    python trainer2.py
```
## Inference
4. 配置测试文件位置和测试模型:
```
    "test":{
            "checkpoint_name":"/home/yangruixiong/ASL2/ASL10/ckpt/epoch60CNNLSTM-mfcc-lfcc.pth",
            "test_data":"path_to_test_file"
    }
```
5. 运行测试程序：
```
    python test.py
```

