import torch.utils.data as data
import numpy as np
from utillsv2 import  Concat_list_all_crop_feedback
import torch
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
from tqdm import tqdm
import option

args = option.parser.parse_args()


class Dataset_Con_all_feedback_XD(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal


        if test_mode:

            self.con_all = Concat_list_all_crop_feedback(True)
        else:
      
            #self.con_all = np.load("concatenated/concat_UCF.npy")
            self.con_all = np.memmap('concat_UCF.npy', dtype='float32', mode='r', shape=(1610, 32, 10, 2048)).copy()
            # self.con_all = np.load("concatenated/concat_XD.npy")
            self.con_all = np.array(self.con_all).reshape(1610, 32, 10, 2048)

            print('self.con_all shape:',self.con_all.shape)

        self.tranform = transform
        self.test_mode = test_mode
        

    def __getitem__(self, index):
        
        
        if self.test_mode:
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)

        else:
            
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)
    



        if self.test_mode:
            
            return features
        else:

            return features , index



    def __len__(self):
        return len(self.con_all)   



class UCFTrainSnippetDataset(data.Dataset):
    def __init__(self, conall_path, pseudo_path):
        # pseudo_path로 길이(sumT)만 맞추려는 용도 (실제 라벨은 original_labels에서 idx로 뽑음)
        self.pseudo = np.load(pseudo_path).astype(np.float32)   # (sumT,)
        self.total_T = self.pseudo.shape[0]
        self.con_all = np.memmap(conall_path, dtype="float32", mode="r",
                                 shape=(self.total_T, 10, 2048))

    def __len__(self):
        return self.total_T

    def __getitem__(self, idx):
        #x = np.array(self.con_all[idx], dtype=np.float32)  # (10,2048)
        #x = x.mean(axis=0)                                 # (2048,)
        x = torch.from_numpy(self.con_all[idx].copy())
        x = x.mean(dim=0)
        #x = torch.from_numpy(x)                            # CPU float32

        # idx는 original_labels(=CUDA tensor)에서 바로 인덱싱 되도록 CUDA LongTensor로 반환 (num_workers=0이면 안전)
        #soft = torch.tensor(self.pseudo[idx], dtype=torch.float32)


        return x, idx


class UCFTestVideoDataset(data.Dataset):
    def __init__(self, conall_path, nalist_path):
        self.nalist = np.load(nalist_path)                 # (290,2)
        self.total_T = int(self.nalist[-1, 1])
        print("total_T from code:", int(self.nalist[-1,1]))
        self.con_all = np.memmap(conall_path, dtype="float32", mode="r",
                                 shape=(self.total_T, 10, 2048))

    def __len__(self):
        return len(self.nalist)

    def __getitem__(self, index):
        a, b = map(int, self.nalist[index])
        x = np.array(self.con_all[a:b], dtype=np.float32)  # (T,10,2048)
        x = x.mean(axis=1)                                 # (T,2048) crop mean
        return torch.from_numpy(x)                         # CPU float32
