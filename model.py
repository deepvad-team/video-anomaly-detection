import torch
import torch.nn as nn
import torch.nn.init as torch_init
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Model(nn.Module):
    def _init_(self, n_features):
        super(Model, self)._init_()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        # x = self.dropout(x)
        return x




class Model_V2(nn.Module): # multiplication then Addition
    def __init__(self, n_features):
        super(Model_V2, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)

        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim = 1))

        self.fc2 = nn.Linear(512, 32)

        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim = 1))

        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()

        self.softplus = nn.Softplus()
        self.apply(weight_init)


    def forward(self, inputs):
        # inputs: (B, T, 2048) OR (B,2048)
        orig_shape = inputs.shape

        if inputs.dim() == 3: #4
            B, T, D = inputs.shape
            x = inputs.reshape(B*T, D) #(B*T, 2048)
            #inputs = inputs.mean(dim=2)  # crop(10) 평균 -> (B, 32, 2048)
        else:
            x = inputs

        #x = self.fc1(inputs)
        att1 = self.fc_att1(x)
        x = self.fc1(x)
        x = (x * att1) + x
        x = self.relu(x)
        x = self.dropout(x)
        

        att2 = self.fc_att2(x)
        x = self.fc2(x)
        x = (x * att2) + x
        x = self.relu(x)
        x = self.dropout(x)

        x = self.softplus(self.fc3(x))

        #지금 32개의 segment 축 평균을 계산해서 segment level anomaly score 가 아닌 (128,10,1) crop level 점수를 내고있었음.
        #x = x.mean(dim = 1) 

        if len(orig_shape) == 3:
            x = x.reshape(B,T,2)

        return x