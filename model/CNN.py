from torch import nn

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=16,
                kernel_size=2,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 8, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

"""
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=12,out_channels=18,kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(18,32,2),
            # torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32,16,2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16,8,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8,4,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(4,2,1),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1),
        )
    def forward(self, x):
        x = self.model(x)
        return x

"""

"""
from torch import nn

class CNNNetwork(nn.Module):

    def __init__(self):
        super(CNNNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=16,
                kernel_size=2,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 4, 2),
            nn.Softmax(dim=1)
        )
        

    def forward(self, x):
        x = self.model(x)
        return x

"""