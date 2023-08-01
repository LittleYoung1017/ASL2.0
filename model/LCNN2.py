from torch import nn
import utils
class LCNN(nn.Module):
#padding 无所谓
#channel num参数减小

    def __init__(self):
        super(LCNN,self).__init__()
        # self.hps = utils.get_hparams_from_file('config.json')
        self.conv1 = nn.Sequential(
            nn.Conv2d(          #input(1,w,h)
                in_channels=1,  #output(128,w,h)          
                out_channels=32,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#output(128,w/2,h/2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(          #input(128,w/2,h/2)
                in_channels=32,  #output(64,w/2,h/2)          
                out_channels=16,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#output(64,w/4,h/4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,   #output(64,w/4,h/4)
                out_channels=8,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  #output(64,w/8,h/8)
            nn.Dropout(0.25),
            nn.Flatten(),
        )
        self.lin = nn.Sequential(
            nn.Linear( 560,8),   #output(64*w/8*h/8,32)  #做pooling
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
        # print("x=========")
        # print(x.shape)
        x = self.lin(x)
        # print("output==========")
        x = x.view(x.size(0), -1)
        # print(x.shape)        
        output = self.output(x)
        # print(output.shape)
        # print(output)
        return output
