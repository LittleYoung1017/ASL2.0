import os 
import numpy as np
import torch
import argparse
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader,Dataset

from utils import MyDataset
from model.CNN import CNN
from model.LCNN2 import LCNN
from model.CNNLSTM2 import CNNLSTM2
import utils
from tqdm import tqdm
from MyDataloader import MyIterableDataset
class Trainer:
    def __init__(self, hps):
        self.hps = hps
        self.device = (
            hps.train.device
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.logger = utils.get_logger(hps.train.log_dir)
        self.model = self.get_model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hps.train.learning_rate)

    def get_model(self):
        model_name = self.hps.model.model_name
        if model_name == 'LCNN':
            model = LCNN().to(self.device)
        elif model_name == 'CNN':
            model = CNN().to(self.device)
        elif model_name == 'CNNLSTM2':
            model = CNNLSTM2(32,32,2,3).to(self.device)
        else:
            raise Exception(f"Error: {model_name}")
        return model

    def load_data(self):
        train_data_x, train_data_y, test_data_x, test_data_y = utils.load_data_new(self.hps)

        training_data = MyDataset(train_data_x, train_data_y)
        test_data = MyDataset(test_data_x, test_data_y)
        return training_data, test_data

    def train(self, dataloader, epoch):
        self.model.train()
        # size = len(dataloader.dataset)
        correct = 0
        loss_sum = []
        correct_sum = []
        acc_sum = []
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(self.device), y.to(self.device)
            X = X.type(torch.cuda.FloatTensor)
            pred = self.model(X)

            correct = (pred.argmax(1) == y).type(torch.float).sum().item() #计算每个batch中预测与结果相同的个数
            acc = correct / len(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            correct_sum.append(correct)
            acc_sum.append(acc)
            loss_sum.append(loss)
            
            if batch % self.hps.train.log_interval == 0:
                loss, current = sum(loss_sum) / len(loss_sum) , (batch + 1) * len(X)
                accuracy = sum(acc_sum) / len(acc_sum)    
                # self.logger.info(f"loss: {loss:>7f}  accuracy: {accuracy*100:>5f} [{current:>5d}/{size:>5d}]")
                self.logger.info(f"loss: {loss:>7f}  accuracy: {accuracy*100:>5f} [{current:>5d}]")

        loss = sum(loss_sum) / len(loss_sum)
        accuracy = sum(acc_sum) / len(acc_sum)
        self.logger.info(f"Train  Loss: {loss:>7f} accuracy: {(accuracy*100):>4f}% ")
        return loss,accuracy

    def val(self, dataloader, epoch):
        self.model.eval()
        # size = len(dataloader.dataset)
        # num_batches = len(dataloader)
        
        val_loss_sum, val_correct_sum = [], []
        val_acc_sum = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                # y = y.reshape(-1,1)
                X = X.type(torch.cuda.FloatTensor)
                pred = self.model(X)

                val_loss = self.loss_fn(pred, y).item()
                val_correct = (pred.argmax(1) == y).type(torch.float).sum().item() #tensor.argmax(dim=-1)每一行最大值下标
                val_acc = val_correct / len(X)
                
                val_loss_sum.append(val_loss)
                val_correct_sum.append(val_correct)
                val_acc_sum.append(val_acc)

        val_loss = sum(val_loss_sum) / len(val_loss_sum)
        val_acc = sum(val_acc_sum) / len(val_acc_sum)
        self.logger.info(f"Val loss: {val_loss:>8f} Val accuracy: {(val_acc*100):>5f}%")
        return val_loss,val_acc

    def run(self):
        self.logger.info(f"Using {self.device} device")
        self.logger.info(f"Start training with {self.hps.model.model_name} model")  
        self.logger.info(self.model)
        
        # from thop import profile
        # x = torch.randn(1,1,32,300).to(self.device)
        # macs, params = profile(self.model, inputs = (x,))
        # self.logger.info('Mults.', macs/1e6,'[M] Params.',params/1e3,'[kB]')
        
        # self.logger.info(f"Config: {self.hps}")
        train_dataset = MyIterableDataset(self.hps.data.downsample_data,'train')
        val_dataset = MyIterableDataset(self.hps.data.downsample_data,'test')
        train_dataloader = DataLoader(train_dataset, batch_size=self.hps.train.batch_size, 
                                      num_workers=16,pin_memory=True,drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.hps.train.batch_size, 
                                    num_workers=16,pin_memory=True,drop_last=False)
        for epoch in range(self.hps.train.epoch):
            self.logger.info(f"====> Epoch: {epoch + 1}")


            loss, acc = self.train(train_dataloader, epoch)
            val_loss, val_acc = self.val(val_dataloader, epoch)

            if epoch and epoch % self.hps.train.ckpt_interval == 0 :
                os.makedirs('ckpt',exist_ok=True)
                ckpt_name = f"epoch{epoch}{self.hps.model.model_name}-MFCC-LFCC.pth"
                self.logger.info(f"Saving checkpoint: {ckpt_name}")
                torch.save(self.model.state_dict(), os.path.join(self.hps.train.checkpoint_path, ckpt_name))

        self.logger.info("Done")


def main():
    config_path = "config.json"
    hps = utils.get_hparams_from_file(config_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="",help="type of the function")
    args = parser.parse_args()
    
    if args.data_path != "":
        utils.modify_config(['data','downsample_data'],args.data_path)  #训练数据路径
    
    trainer = Trainer(hps)
    trainer.run()


if __name__ == "__main__":
    main()

    