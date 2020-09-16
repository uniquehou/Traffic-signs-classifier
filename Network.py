from torch import nn
import os
from Config import Config

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.config = Config.copy()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48*2, 11, stride=4, padding=0),
            nn.BatchNorm2d(48*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48*2, 128*2, 5, 1, 2),
            nn.BatchNorm2d(128*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128*2, 192*2, 3, 1, 1),
            nn.BatchNorm2d(192*2),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192*2, 192*2, 3, 1, 1),
            nn.BatchNorm2d(192*2),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192*2, 128*2, 3, 1, 1),
            nn.BatchNorm2d(128*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6*6*128*2, 2048*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048*2, 2048*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048*2, 62)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 6*6*128*2)
        x_class = self.classifier(x)
        return x_class

    def getConfig(self):
        self.config['train_size'] = 227
        self.config['train_batch_size'] = 200
        if not os.path.exists(self.config['model_dir']):
            os.mkdir(self.config['model_dir'])
        return self.config

class Net(nn.Module):
    def __init__(self):
        self.config = Config.copy()
        # input: 128*128*3
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),    # 126*126*6
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)    # 63*63*6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, 3),    # 61*61*12
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3),    # 59*59*12
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2),    # 29*29*12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 24, 3),    # 27*27*24
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3),    # 25*25*24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 12*12*24
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 48, 3),    # 10*10*48
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3),    # 3*3*48
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)    # 4*4*48
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*4*48, 512),
            # nn.Dropout2d(),
            nn.Linear(512, 128),
            # nn.Dropout2d(),
            nn.Linear(128, 62)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4*4*48)
        x_class = self.classifier(x)
        return x_class

    def getConfig(self):
        self.config['train_size'] = 128
        self.config['model_dir'] = 'best_net_wts.pt'
        if not os.path.exists(self.config['model_dir']):
            os.mkdir(self.config['model_dir'])
        return self.config


class PlusNet(nn.Module):
    def __init__(self):
        self.config = Config.copy()
        # input: 128*128*3
        super(PlusNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),    # 126*126*6
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)    # 63*63*6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 20, 3),    # 61*61*12
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 20, 3),    # 59*59*12
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2),    # 29*29*12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 50, 3),    # 27*27*24
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 50, 3),    # 25*25*24
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 12*12*24
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 120, 3),    # 10*10*48
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 120, 3),    # 3*3*48
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)    # 4*4*48
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*4*120, 512),
            nn.Dropout2d(),
            nn.Linear(512, 128),
            nn.Dropout2d(),
            nn.Linear(128, 62)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4*4*120)
        x_class = self.classifier(x)
        return x_class

    def getConfig(self):
        self.config['train_size'] = 128
        self.config['best_wts_name'] = 'plus_net_best_wts.pt'
        if not os.path.exists(self.config['model_dir']):
            os.mkdir(self.config['model_dir'])
        return self.config


class StrongNet(nn.Module):
    def __init__(self):
        self.config = Config.copy()
        super(StrongNet, self).__init__()
        # input: 200*200*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, 3, 1),    # 198*198*12
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 99*99*12
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 50, 3, 1),    # 97*97*50
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 50, 3, 1, 1),    # 97*97*50
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 48*48*50
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 200, 3, 1),    # 46*46*200
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 200, 3, 1),    # 44*44*120
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 200, 3, 1),   # 42*42*120
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 21*21*200
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(200, 500, 3, 1),    # 19*19*500
            nn.ReLU(inplace=True),
            nn.Conv2d(500, 500, 3, 1),    # 17*17*500
            nn.ReLU(inplace=True),
            nn.Conv2d(500, 500, 3, 1, 1),    # 17*17*500
            nn.BatchNorm2d(500),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 8*8*500
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(500, 1000, 3, 1),    # 6*6*1000
            nn.ReLU(inplace=True),
            nn.Conv2d(1000, 1000, 3, 1, 1),    # 6*6*1000
            nn.ReLU(inplace=True),
            nn.Conv2d(1000, 1000, 3, 1, 1),    # 6*6*1000
            nn.BatchNorm2d(1000),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 3*3*1000
        )
        self.classifier = nn.Sequential(
            nn.Linear(3*3*1000, 2048),
            nn.Dropout2d(),
            nn.Linear(2048, 1024),
            nn.Dropout2d(),
            nn.Linear(1024, 512),
            nn.Dropout2d(),
            nn.Linear(512, 62),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 3*3*1000)
        x_class = self.classifier(x)
        return x_class

    def getConfig(self):
        self.config['train_size'] = 200
        self.config['model_dir'] = 'trained_StrongNet_model'
        self.config['best_wts_name'] = 'best_StrongNet_wts.pt'
        if not os.path.exists(self.config['model_dir']):
            os.mkdir(self.config['model_dir'])
        return self.config

