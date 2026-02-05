python test.py \
    --lossfn BCE \
    --optimizer SGD \
    --max-epoch 20 \
    --outer-epochs 1\
    --batch-size 1\
    --lr 0.001\
    --feature-size 2048\
    --gt list/gt-ucf-RTFM.npy\
    --datasetname UCF \
    --windowsize 0.15\
    --eps 0.225 \
    --eps2 1.32475  \
    --pseudofile Unsup_labels/UCF_hard_label.npy\
    --conall concat_UCF\
    #--create True\