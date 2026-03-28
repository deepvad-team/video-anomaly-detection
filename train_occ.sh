python mainv2.py \
  --train-mode wbce \
  --optimizer SGD \
  --max-epoch 20\
  --batch-size 128 \
  --lr 0.01 \
  --feature-size 2048 \
  --gt list/gt-ucf-RTFM.npy \
  --datasetname UCF \
  --pseudofile pseudo_prop2_label_glist_rw.npy \
  --pseudo-label-file pseudo_prop2_label_glist_rw.npy \
  --pseudo-weight-file pseudo_prop2_weight_glist_rw.npy\
  --normal-source pseudo \
  --normal-thr 0.3 \
  --lambda-compact 1.0 \
  --lambda-bce 0.0
  #--eval-every 0