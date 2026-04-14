import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, return_logits=False, return_feats=False): #변경 (추가) 부분: return_logits 받음

        #x = self.fc1(inputs)
        att1 = self.fc_att1(inputs)
        x = self.fc1(inputs)
        x = (x * att1) + x
        x = self.relu(x)
        x = self.dropout(x)

        att2 = self.fc_att2(x)
        x = self.fc2(x)
        x = (x * att2) + x
        feat = self.relu(x)     # <- 2번째 relu 통과 후 32차원 벡터를 compact loss 계산용 feature 로 추출
        x = self.dropout(feat)

        #변경 (추가) 부분: sigmoid 통과 전 logits 도 반환
        logits = self.fc3(x)  # (N, 1) OR (N, 10, 1) depending on input
        prob = self.sigmoid(logits) 

        if prob.dim() > 2:
            prob_out = prob.mean(dim=1)  #10crop이면 평균내주기
            logits_out = logits.mean(dim=1)
            feat_out = feat.mean(dim=1)
        else:
            prob_out = prob
            logits_out = logits
            feat_out = feat
        
        outputs = [prob_out]

        if return_logits and return_feats:
            return prob_out, logits_out, feat_out
        elif return_logits:
            return prob_out, logits_out
        elif return_feats:
            return prob_out, feat_out
        else:
            return prob_out 


class Model_V2_AllCNN(nn.Module):
    
    def __init__(self, n_features, kernel_size=5):
        super().__init__()
        
        # Layer 1: 2048 → 512 (WITH temporal!)
        self.conv1 = nn.Conv1d(n_features, 256, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Attention (also temporal!)
        self.conv_att1 = nn.Conv1d(n_features, 256, kernel_size, padding=kernel_size//2)
        
        # Layer 2: 512 → 128 (WITH temporal!)
        self.conv2 = nn.Conv1d(256, 64, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Attention (also temporal!)
        self.conv_att2 = nn.Conv1d(256, 64, kernel_size, padding=kernel_size//2)
        
        # Layer 3: 128 → 32 (WITH temporal!)
        #self.conv3 = nn.Conv1d(128, 32, kernel_size, padding=kernel_size//2)
        #self.bn3 = nn.BatchNorm1d(32)
        
        # Output: Only 1 FC at the end
        self.fc_out = nn.Linear(64, 1)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs, return_logits=False):
        orig_shape = inputs.shape
        
        if inputs.dim() == 3:
            B, T, D = inputs.shape
            use_temporal = True
        else:
            B = inputs.shape[0]
            T = 1
            inputs = inputs.unsqueeze(1)
            use_temporal = False
        
        # Permute for Conv1d: (B, T, D) → (B, D, T)
        x = inputs.permute(0, 2, 1)  # (B, 2048, T)
        
        att1 = torch.sigmoid(self.conv_att1(x))  # (B, 512, T)
        x = self.conv1(x)                         # (B, 512, T)
        #x = self.bn1(x)
        x = x * att1 + att1  # Gated attention
        x = self.gelu(x)
        x = self.dropout1(x)
        
        att2 = torch.sigmoid(self.conv_att2(x))  # (B, 32, T)
        x = self.conv2(x)                         # (B, 32, T)
        #x = self.bn2(x)
        x = x * att2 + att2
        x = self.gelu(x)
        x = self.dropout2(x)
        
        #x = self.conv3(x)      # (B, 32, T)
        #x = self.bn3(x)
        #x = self.relu(x)
        
        # Back to (B, T, 32)
        x = x.permute(0, 2, 1)
        
        # Output FC
        if use_temporal and T > 1:
            x = x.reshape(B, T, 64)
        else:
            x = x.squeeze(1)
        
        logits = self.fc_out(x)              # raw logits
        prob = torch.sigmoid(logits)         # raw prob

        #x = self.sigmoid(self.fc_out(x))

        prob_s = prob.squeeze(-1) # (B, T)

        if prob_s.dim() == 2:                # (B,T)
            prob_s = F.avg_pool1d(prob_s.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
            prob_s = prob_s.unsqueeze(-1)    # (B,T,1)
        else:
            prob_s = prob                    # single-step fallback

        if return_logits:
            return prob_s, logits
        return prob_s
