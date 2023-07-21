from torch import nn
import utils
class LCNN(nn.Module):

    def __init__(self):
        super(LCNN,self).__init__()
        # self.hps = utils.get_hparams_from_file('config.json')
        self.conv1 = nn.Sequential(
            nn.Conv2d(          #input(1,w,h)
                in_channels=1,  #output(128,w,h)          
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#output(128,w/2,h/2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(          #input(128,w/2,h/2)
                in_channels=128,  #output(64,w/2,h/2)          
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#output(64,w/4,h/4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,   #output(64,w/4,h/4)
                out_channels=64,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  #output(64,w/8,h/8)
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear( 640,32),   #output(64*w/8*h/8,32)
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(32,2),
            # nn.Softmax(dim=0)
            # nn.Linear(2,2),
            # nn.Softmax(dim=1)    
        )
            
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print("x=========")
        # print(x.shape)
        # print("output==========")
        x = x.view(x.size(0), -1)        
        output = self.output(x)
        # print(output)
        return output,x
