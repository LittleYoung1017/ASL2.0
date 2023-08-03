from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import get_worker_info
from torch.utils import data
import numpy as np
import librosa
import os
import time
import multiprocessing
class MyIterableDataset(IterableDataset):
    def __init__(self, file_path,data_type):
        super(MyIterableDataset, self).__init__()
        self.file_path = file_path # Only load file path
        self.data_type = data_type
    def load_data(self):
        set_list = os.listdir(self.file_path)
        for dataset in set_list:
            data_path = os.path.join(self.file_path,dataset,'feature_data')
            file_list = os.listdir(data_path)
            if self.data_type =='train':
                file_list = [i for i in file_list if 'train' in i]
            elif self.data_type == 'test':
                file_list = [i for i in file_list if 'test' in i]
            for f in file_list:
                data = np.load(os.path.join(data_path,f))
                X_data = data['matrix']
                X_data = X_data[:,np.newaxis,:,:]
                y_data = data['labels']
                for i in range(len(X_data)):
                    yield X_data[i], y_data[i]
    def load_data_multi(self,worker_info):
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        set_list = os.listdir(self.file_path)
        for i, dataset in enumerate(set_list):
            if i % num_workers == worker_id:
                data_path = os.path.join(self.file_path, dataset, 'feature_data')
                file_list = os.listdir(data_path)
                if self.data_type =='train':
                    file_list = [i for i in file_list if 'train' in i]
                elif self.data_type == 'test':
                    file_list = [i for i in file_list if 'test' in i]
                for f in file_list:
                    data = np.load(os.path.join(data_path, f))
                    X_data = data['matrix']
                    X_data = X_data[:,np.newaxis,:,:]
                    y_data = data['labels']
                    for i in range(len(X_data)):
                        yield X_data[i], y_data[i]
                        
    def __iter__(self):
        # Process a single data depending on how the data is read
        worker_info = get_worker_info()
        # print('worker_info:',worker_info)
        if worker_info is None: #single-process data loading
            return self.load_data()
        else: #multi-process data loading
            return self.load_data_multi(worker_info)
        
if __name__ =='__main__':
    start_time = time.time()
    file_path = '/home/yangruixiong/ASL2/data/'
    print(file_path)
    num_workers = multiprocessing.cpu_count()
    print('num_workers:',num_workers)
    
    train_dataset = MyIterableDataset(file_path,'train')
    mydataloader = DataLoader(mydataset,batch_size=500,num_workers=16,pin_memory=True,drop_last=False)
    for X,y in mydataloader:
        print(X.shape)
    end_time = time.time()
    execution_time = end_time - start_time
    print("execution time:",execution_time)