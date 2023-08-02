from torch import nn
import utils
class LCNN(nn.Module):
    def __init__(self):
        super(LCNN,self).__init__()
        # self.hps = utils.get_hparams_from_file('config.json')
        self.conv1 = nn.Sequential(
            nn.Conv2d(         
                in_channels=1,           
                out_channels=32,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(          
                in_channels=32,           
                out_channels=16,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,   
                out_channels=8,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear( 560,8),   
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(8,2),
            # nn.Linear(2,2),
            # nn.Softmax(dim=1)    
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)      
        output = self.output(x)
        return output
