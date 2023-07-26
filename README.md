ASL is a common diagram for audio splicing detection and localization and use a LCNN+LSTM model.


## Data preprocessing
1. 对两个audio数据集分别划分训练集和验证集，将音频数据分割成固定长度音频，两两随机拼接，设置分割长度为3s。再对切割好的音频和拼接生成的音频提取MFCC+LFCC特征，两个特征合并在一起，最终保存为npz文件。
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
生成的的文件目录形式如下：
```
    -dataset1              
        -split_audio
            -train
                -original_1.wav #3s split audio file
                -original_2.wav
                 ...
            -val
                -original_1.wav
                -original_2.wav
                 ...
        -feature_data 
            -train_data_mfcc_lfcc0.npz
            -train_data_mfcc_lfcc1.npz
             ...
            -test_data_mfcc_lfcc0.npz
             ...
    -dataset2
        -split_audio
        -feature_data
    -concat_dataset1_dataset2          /拼接后的数据集
        -split_audio
        -feature_data
```

直接使用asd_data进行训练时，需要将要提取特征的数据集路径保存在config中：
```
    "data":{
        "downsample_data":"path_to_data"
    }
```

## Train
2. 进行少数据量训练：
```
    python trainer.py
```
大数据集训练执行(后续补充)：
```
    python trainer2.py
```
## Inference
3. 配置测试文件位置和测试模型:
```
    "test":{
            "checkpoint_name":"/home/yangruixiong/ASL2/ASL10/ckpt/epoch60CNNLSTM-mfcc-lfcc.pth",
            "test_data":[
                "path_to_test_data1",
                "path_to_test_data2"       
            ]
    }
```
4. 运行测试程序：
```
    python test.py
```

