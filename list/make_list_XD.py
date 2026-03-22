import os
import glob

root_path = r"C:\Users\jplabuser\I3D_Feature_Extraction_resnet\output_nl_XD"    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))
violents = []
normal = []
with open('XD_rgb_test_R50NL.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)