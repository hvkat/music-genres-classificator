import torch.nn as nn
import torch.nn.functional as F

class ClassifierMusicGenres(nn.Module):
    def __init__(self, inputSize, numclasses):
        super(ClassifierMusicGenres,self).__init__()
        self.inputSize = inputSize
        self.fc1=nn.Linear(inputSize,512)
        self.fc2=nn.Linear(512,1024)
        self.fc3=nn.Linear(1024,2048)
        self.fc4=nn.Linear(2048,512)
        self.fc5=nn.Linear(512,256)
        self.fc6=nn.Linear(256,64)
        self.fc7=nn.Linear(64,numclasses)

    def forward(self,x):
        #x = x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=F.softmax(self.fc7(x), dim=1)
        return x

