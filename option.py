import argparse

parser = argparse.ArgumentParser(description='C2FPL')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')

parser.add_argument('--rgb-list', default='./list/ucf-c3d.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default='./list/ucf-c3d-test.list', help='list of test rgb features ')
parser.add_argument('--gt', default='list/gt-ucf_RTFM.npy', help='file of ground truth ')

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
#parser.add_argument('--pseudofile',type=str,  default='UCF_labels_org', help='ground truth file')
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
parser.add_argument('--nalist-path', type=str, default='list/nalist_i3d.npy')
parser.add_argument('--train-concat-path', type=str, default='../../C2FPL/concat_UCF.npy')
parser.add_argument('--test-concat-path', type=str, default='Concat_XD_test_R50NL.npy')
parser.add_argument('--center-path', type=str, default='unsupervised_ckpt/center.pt')

parser.add_argument('--eval-every', type=int, default=0,
                    help='0이면 epoch 끝에서만 평가, >0이면 매 N iteration마다 평가')

parser.add_argument('--model_ckpt', type=str, required=True,
                    help='Path to trained detector checkpoint')

parser.add_argument('--meta_adapter_ckpt', type=str, default=None,
                    help='Path to meta-trained adapter checkpoint')



# tta + maml
parser.add_argument('--conall_path', type=str, default='../../C2FPL/concat_UCF.npy',
                    help='Path to training concat feature file for meta training')

parser.add_argument('--nalist_path_meta', type=str, default='list/nalist_i3d.npy',
                    help='Path to train nalist for meta training')
parser.add_argument('--save_dir', type=str, default='meta_adapter_ckpt',
                    help='Directory to save meta adapter checkpoints')
parser.add_argument('--warmup_segments', type=int, default=5,
                    help='Number of prefix segments used for adaptation')

parser.add_argument('--inner_steps', type=int, default=5,
                    help='Inner adaptation steps on prefix')

parser.add_argument('--inner_lr', type=float, default=1e-3,
                    help='Inner adaptation learning rate')

parser.add_argument('--outer_lr', type=float, default=5e-4,
                    help='Outer meta learning rate')

parser.add_argument('--meta_epochs', type=int, default=10,
                    help='Number of meta training epochs')

parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='Gradient clipping for meta training')

parser.add_argument('--label_smoothing', action='store_true',
                    help='Use label smoothing in suffix uter loss')

parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')


# gate 실험
parser.add_argument("--pseudofile", type=str, required=True)


parser.add_argument("--save_path", type=str, default="prefix_gate_best.pt")
parser.add_argument("--adapter_init_path", type=str, default="adapter_init.pt")

parser.add_argument("--feature_size", type=int, default=2048)

parser.add_argument("--tea_lr", type=float, default=1e-2)
parser.add_argument("--tea_steps_per_video", type=int, default=30)

parser.add_argument("--improve_margin", type=float, default=0.001)

parser.add_argument("--gate_lr", type=float, default=1e-3)
parser.add_argument("--gate_epochs", type=int, default=50)
parser.add_argument("--hidden_dim", type=int, default=16)

#parser.add_argument('--gate_ckpt', type=str, default=None,
#                   help='Path to trained prefix gate checkpoint')

parser.add_argument('--gate_threshold', type=float, default=0.5,
                    help='Threshold for applying prefix TTA')



# one shot prefix
parser.add_argument('--hyper_ckpt', type=str, default=None,
                    help='Path to trained prefix hyper checkpoint')


# meta policy 
parser.add_argument('--policy_ckpt', type=str, default=None,
                    help='Path to trained safe meta policy checkpoint')