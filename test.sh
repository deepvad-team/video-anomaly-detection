python test.py \
    --lossfn BCE \
    --optimizer SGD \
    --max-epoch 20 \
    --outer-epochs 1\
    --batch-size 128 \
    --lr 0.01\
    --feature-size 2048 \
    --gt list/gt-ucf-RTFM.npy\
    --datasetname UCF \
    --pseudofile ../../C2FPL/Unsup_labels/UCF_unsup_labels_original_New_V88_1.5.npy\
    --nalist-path list/nalist_test_i3d.npy
    #--conall concat_XD_test.npy\
    #--nalist-path list/nalist_XD_test.npy
    #--xd_feat concat_XD_test.npy\

