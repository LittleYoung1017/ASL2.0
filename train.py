import os 
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision import datasets
from utils import MyDataset
from model.CNN import CNN
from model.LCNN import LCNN
import utils
from tqdm import tqdm


def load_data(hps):
    train_data_x = np.load(hps.data.prepared_train_data_x)
    train_data_y = np.load(hps.data.prepared_train_data_y)
    print(train_data_x.shape)
    for i in tqdm(range(5)):
        single_data_x = np.load(hps.data.prepared_train_data_x.split('.')[0]+str(i)+'.npy')
        print(single_data_x.shape)
        train_data_x = np.concatenate((train_data_x,single_data_x),axis=0)
        print(train_data_x.shape)
        single_data_y = np.load(hps.data.prepared_train_data_y.split('.')[0]+str(i)+'.npy')
        train_data_y = np.concatenate((train_data_y,single_data_y),axis=0)
        print(train_data_y.shape)
    # train_data_x = np.load(hps.data.prepared_train_data_x)
    train_data_x = train_data_x[:,np.newaxis,:,:]  #for LCNN
    # train_data_y = np.load(hps.data.prepared_train_data_y)
    
    test_data_x = np.load(hps.data.prepared_test_data_x)
    test_data_x = test_data_x[:,np.newaxis,:,:]  #for LCNN
    test_data_y = np.load(hps.data.prepared_test_data_y)


    training_data = utils.MyDataset(train_data_x,train_data_y)
    test_data = utils.MyDataset(test_data_x,test_data_y)
    # return train_data_x, train_data_y, test_data_x, test_data_y
    return training_data, test_data
def train(hps,dataloader,epoch,model,loss_fn,optimizer,logger):
    size = len(dataloader.dataset)
    device = (
        hps.train.device
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.train()
    correct = 0
    loss_sum = []
    correct_sum = []
    acc_sum = []
    for batch, (X, y) in enumerate(dataloader):
        # print(X.shape)
        # print(y.shape)
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)
        pred,_ = model(X)

        correct = (pred.argmax(1) == y).type(torch.float).sum().item() #计算每个batch中预测与结果相同的个数
        acc = correct / len(X)

        loss = loss_fn(pred, y)
        
        
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct_sum.append(correct)
        acc_sum.append(acc)
        loss_sum.append(loss)
        
        if batch % hps.train.log_interval == 0:
            loss, current = sum(loss_sum) / len(loss_sum) , (batch + 1) * len(X)
            accuracy = sum(acc_sum) / len(acc_sum)    
            logger.info(f"loss: {loss:>7f}  accuracy: {accuracy:>5f} [{current:>5d}/{size:>5d}]")

    loss = sum(loss_sum) / len(loss_sum)
    accuracy = sum(acc_sum) / len(acc_sum)
    logger.info(f"Train accuracy: {(accuracy*100):>4f}% Loss: {loss:>7f}")
    return loss,accuracy

def val(hps,dataloader,epoch, model, loss_fn,logger):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = (
        hps.train.device
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.eval()
    val_loss_sum, val_correct_sum = [], []
    val_acc_sum = []
    with torch.no_grad():
        for X, y in dataloader:
            y_temp = y.numpy()
            X, y = X.to(device), y.to(device)
            # y = y.reshape(-1,1)
            X = X.type(torch.cuda.FloatTensor)
            pred,_ = model(X)

            val_loss = loss_fn(pred, y).item()
            val_correct = (pred.argmax(1) == y).type(torch.float).sum().item() #tensor.argmax(dim=-1)每一行最大值下标
            val_acc = val_correct / len(X)
            
            val_loss_sum.append(val_loss)
            val_correct_sum.append(val_correct)
            val_acc_sum.append(val_acc)

    val_loss = sum(val_loss_sum) / len(val_loss_sum)
    val_acc = sum(val_acc_sum) / len(val_acc_sum)
    logger.info(f"Val Accuracy: {(val_acc):>5f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss,val_acc

def test(hps,dataloader,epoch, model, loss_fn,logger):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = (
        hps.train.device
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.eval()
    test_loss_sum, test_correct_sum = [], []
    test_acc_sum = []
    with torch.no_grad():
        for X, y in dataloader:
            y_temp = y.numpy()
            X, y = X.to(device), y.to(device)
            # y = y.reshape(-1,1)
            X = X.type(torch.cuda.FloatTensor)
            pred,_ = model(X)

            test_correct = (pred.argmax(1) == y).type(torch.float).sum().item() #tensor.argmax(dim=-1)每一行最大值下标
            test_acc = test_correct / len(X)
            
            test_loss_sum.append(test_loss)
            test_correct_sum.append(test_correct)
            test_acc_sum.append(test_acc)

    test_loss = sum(test_loss_sum) / len(test_loss_sum)
    test_acc = sum(test_acc_sum) / len(test_acc_sum)
    logger.info(f"Val Accuracy: {(test_acc):>5f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss,test_acc

def main():
    config_path = 'config.json'
    hps = utils.get_hparams_from_file(config_path)
    logger = utils.get_logger(hps.train.log_dir)
    model_name = hps.model.model_name
    
    train_data_x = np.load(hps.data.prepared_train_data_x)
    train_data_y = np.load(hps.data.prepared_train_data_y)
    print(train_data_x.shape)
    for i in tqdm(range(5)):
        single_data_x = np.load(hps.data.prepared_train_data_x.split('.')[0]+str(i)+'.npy')
        print(single_data_x.shape)
        train_data_x = np.concatenate((train_data_x,single_data_x),axis=0)
        print(train_data_x.shape)
        single_data_y = np.load(hps.data.prepared_train_data_y.split('.')[0]+str(i)+'.npy')
        train_data_y = np.concatenate((train_data_y,single_data_y),axis=0)
        print(train_data_y.shape)
    # train_data_x = np.load(hps.data.prepared_train_data_x)
    train_data_x = train_data_x[:,np.newaxis,:,:]  #for LCNN
    # train_data_y = np.load(hps.data.prepared_train_data_y)
    
    test_data_x = np.load(hps.data.prepared_test_data_x)
    test_data_x = test_data_x[:,np.newaxis,:,:]  #for LCNN
    test_data_y = np.load(hps.data.prepared_test_data_y)


    batch_size = hps.train.batch_size
    training_data = utils.MyDataset(train_data_x,train_data_y)
    test_data = utils.MyDataset(test_data_x,test_data_y)
    # train_dataloader = DataLoader(data, batch_size=200, shuffle=False, drop_last=True, num_workers=0)


    
    device = (
        hps.train.device
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    if model_name == 'LCNN':
        model = LCNN().to(device)
    elif model_name =='CNN':
        model = CNN().to(device)
    elif model_name =='CNNLSTM':
        model = CNNLSTM().to(device)
    else:
        raise Exception('Error:{}'.format(model_name))
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=hps.train.learning_rate)
    epochs = hps.train.epoch
    logger.info(f'=========Start training with {hps.model.model_name} model==========')
    logger.info(f'config: {hps}')
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1}\n-------------------------------")


        logger.info('====> Epoch: {}'.format(epoch+1))
        train_dataloader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=True)
        
        loss,acc = train(hps,train_dataloader, epoch, model, loss_fn, optimizer,logger)
        val_loss,val_acc = val(hps,test_dataloader, epoch, model, loss_fn,logger)

        if epoch % hps.train.ckpt_interval == 0:
            ckpt_name = 'epoch'+ str(epoch) + hps.model.model_name + '.pth'
            # print(ckpt_name)
            logger.info(f'Saving checkpoint: {ckpt_name}')
            torch.save(model.state_dict(),os.path.join(hps.train.checkpoint_path,ckpt_name))
    logger.infor("Done")
    
if __name__ == '__main__':
    main()
        
