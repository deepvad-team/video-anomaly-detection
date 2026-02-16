python test.py \
    --lossfn BCE \
    --optimizer SGD \
    --max-epoch 20 \
    --outer-epochs 1\
    --batch-size 128 \
    --lr 0.01\
    --feature-size 2048 \
    --gt ../../C2FPL/list/gt-ucf-RTFM.npy\
    --datasetname UCF \
    --windowsize 0.15\
    --eps 0.225 \
    --eps2 1.32475  \
    --pseudofile ../../C2FPL/Unsup_labels/UCF_unsup_labels_original_New_V88_1.5.npy\
    --conall concat_UCF\
 