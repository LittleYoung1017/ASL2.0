from torch.utils.data import IterableDataset, DataLoader
from torch.utils import data
import numpy as np
import librosa
class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        super(MyIterableDataset, self).__init__()
        self.file_path = file_path # Only load file path
    
    def parse_feature(self):
        file_list = os.listdir(file_path)
        for f in file_list:
            data = np.load(f)
            X_data = data['matrix']
            y_data = data['labels']
            for i in range(len[i]):
                yield X_data[i], y_data[i]
    def __iter__(self):
         # Process a single data depending on how the data is read
        return self.parse_feature()
        

    
if __name__ =='__main__':
    file_path = '/home/yangruixiong/ASL2/data/concat-ESC-3s-8k/feature_data'
    print(file_path)
    mydataset = MyIterableDataset(file_path)
    mydataloader = DataLoader(mydataset,batch_size=1000,drop_last=True)
    for data in mydataloader:
        print(data.shape)