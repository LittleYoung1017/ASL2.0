import torch
import torch.nn as nn
import numpy as np
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, lstm_output):
        attention_scores = self.attention_weights(lstm_output)
        attention_weights = self.softmax(attention_scores)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size)  # 卷积层
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size)  # 卷积层
        self.pool = nn.MaxPool1d(kernel_size=2)  # 池化层
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size)  # 卷积层 
        self.conv4 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size)  # 卷积层
        self.pool = nn.MaxPool1d(kernel_size=2)  # 池化层
        self.lstm1 = nn.LSTM(hidden_dim//2, hidden_dim//2, batch_first=True)  # LSTM层
        self.lstm2 = nn.LSTM(hidden_dim//2, hidden_dim//4, batch_first=True)  # LSTM层
        self.attention = Attention(hidden_dim//4)
        self.fc1 = nn.Linear(hidden_dim//4, hidden_dim//8)  # 全连接层
        self.fc2 = nn.Linear(hidden_dim//8, output_dim)  # 全连接层
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer with dropout rate of 0.5

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 卷积层激活函数 
        x = torch.relu(self.conv2(x))  # 卷积层激活函数 
        x = self.pool(x)  # 池化层 
        x = torch.relu(self.conv3(x))  # 卷积层激活函数 
        x = torch.relu(self.conv4(x))  # 卷积层激活函数 
        x = self.pool(x)  # 池化层 
        

        x = x.permute(0, 2, 1)  # 调整维度顺序以适应LSTM输入
        lstm_output1, _ = self.lstm1(x)  # LSTM层
        lstm_output2, _ = self.lstm2(lstm_output1)  # LSTM层
        context_vector = self.attention(lstm_output2)
        x = torch.relu(self.fc1(context_vector))  # 全连接层激活函数
        x = self.dropout(x)  #暂时添加dropout
        x = self.fc2(x)  # 全连接层
        return x

