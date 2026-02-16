from matplotlib.pyplot import axis
import numpy as np
import torch.utils.data as data
import pandas as pd
import option
from scipy.stats import multivariate_normal
import os
import matplotlib.pyplot as plt
import torch


args = option.parser.parse_args()


def Concat_list_all_crop_feedback(Test=False, create='False'): #UCF
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    if Test is True:
        gt = np.load(args.gt)
        #변경부분--
        # con_test = np.load("concatenated/Concat_test_10.npy")
        total_T = len(gt) // 16 
        con_test = np.memmap('../../C2FPL/Concat_test_10.npy',dtype="float32",mode="r",shape=(total_T,10,2048)).copy() #(69634,10,2048)

        print('Testset size:', con_test.shape)
        return con_test
    
    if Test is False:
        #현재 create args를 받지는 않고있음. (즉, 32 33은 실행되지 않는 줄)
        if create == 'True':
            print('loading Pseudo Labels......',args.pseudofile)

        label_all = np.load(args.pseudofile)
        print('[*] concatenated labels shape:',label_all.shape) #(779951,)
        return len(label_all), torch.tensor(label_all).cuda()

