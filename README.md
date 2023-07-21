ASL is a common diagram for audio splicing detection and localization.
this is a LCNN+LSTM model.


## Data preprocessing
```
python downsampling_mfcc_lfcc_concat2.py
```

## Train
```
python trainer.py
```

## inference
```
python test.py
```


数据多线程加载：

'''
import threading
 # 定义数据加载线程
class DataLoaderThread(threading.Thread):
    def __init__(self, data, batch_size):
        threading.Thread.__init__(self)
        self.data = data
        self.batch_size = batch_size
     def run(self):
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i+self.batch_size]
            # 将batch_data加载到模型中进行训练
            # 进行训练操作...
 # 定义训练线程
class TrainingThread(threading.Thread):
    def __init__(self, num_epochs):
        threading.Thread.__init__(self)
        self.num_epochs = num_epochs
     def run(self):
        for epoch in range(self.num_epochs):
            # 创建数据加载线程并启动
            data_loader_thread = DataLoaderThread(data, batch_size)
            data_loader_thread.start()
             # 进行训练操作...
            # 等待数据加载线程完成
            data_loader_thread.join()
 # 设置参数
data = [...]  # 数据集
batch_size = 32  # 每个batch的大小
num_epochs = 10  # 迭代的epoch数
 # 创建训练线程并启动
training_thread = TrainingThread(num_epochs)
training_thread.start()
 # 等待训练线程完成
training_thread.join()
'''