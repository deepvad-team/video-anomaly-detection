import argparse

parser = argparse.ArgumentParser(description='C2FPL')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')

parser.add_argument('--rgb-list', default='./list/ucf-c3d.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default='./list/ucf-c3d-test.list', help='list of test rgb features ')
parser.add_argument('--gt', default='list/gt-ucf-RTFM.npy', help='file of ground truth ')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.0001)')

parser.add_argument('--batch-size', type=int, default=128, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=0, help='number of workers in dataloader')
parser.add_argument('--model-name', default='C2FPL', help='name to save model')

parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--datasetname', default='UCF', help='dataset to train on (default: )')

parser.add_argument('--max-epoch', type=int, default=30, help='maximum iteration to train (default: 100)')

parser.add_argument('--optimizer', default='SGD', help='Number of segments of each video')
parser.add_argument('--lossfn', default='BCE', help='Number of segments of each video')
parser.add_argument('--stepsize',type=int,  default=5, help='lr_scheduler stepsize')
parser.add_argument('--pseudo',type=str,  default='Unsup_labels/UCF_labels_org.npy', help='pseudo labels')


parser.add_argument('--windowsize',type=float,  default=0.09, help='lr_scheduler stepsize')
parser.add_argument('--modelversion',type=str,  default='Model_V2', help='Model version')
parser.add_argument('--eps2',type=float,  default=0.4, help='lr_scheduler stepsize')
parser.add_argument('--outer-epochs',type=int,  default=1, help='lr_scheduler stepsize')
parser.add_argument('--pseudofile',type=str,  default='UCF_labels_org', help='ground truth file')
parser.add_argument('--conall',type=str,  default='concat_UCF', help='ground truth file')


parser.add_argument('--xd_feat', type=str, help='path to XD feature npy, shape=(145649,1024)'
)

# 0323 추가 --------------------------------------

# mode
parser.add_argument('--train-mode', type=str, default='bce',
                    choices=['bce', 'occ', 'hybrid', 'wbce'])
parser.add_argument('--normal-source', type=str, default='pseudo',
                    choices=['pseudo', 'prefix', 'intersect'])
parser.add_argument('--pseudo-label-file', type=str, default='')
parser.add_argument('--pseudo-weight-file', type=str, default='')
# normal selection
parser.add_argument('--normal-thr', type=float, default=0.3,
                    help='pseudo score <= thr 를 normal 로 사용')
parser.add_argument('--prefix-len', type=int, default=3)

# occ loss
parser.add_argument('--lambda-compact', type=float, default=1.0)
parser.add_argument('--lambda-bce', type=float, default=0.0)

# inference score
parser.add_argument('--score-mode', type=str, default='distance',
                    choices=['prob', 'distance', 'mix'])
parser.add_argument('--mix-alpha', type=float, default=0.5)

# paths
parser.add_argument('--nalist-path', type=str, default='list/nalist_XD_test_R50NL.npy')
parser.add_argument('--train-concat-path', type=str, default='../../C2FPL/concat_UCF.npy')
parser.add_argument('--test-concat-path', type=str, default='Concat_XD_test_R50NL.npy')
parser.add_argument('--center-path', type=str, default='unsupervised_ckpt/center.pt')

parser.add_argument('--eval-every', type=int, default=0,
                    help='0이면 epoch 끝에서만 평가, >0이면 매 N iteration마다 평가')

