# bridge_api.py
from resnet import i3_res50

def load_i3d_model(pretrainedpath):
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.eval()   # train(False)와 같은 의미지만 이게 더 명확
    return i3d