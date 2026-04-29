python test.py \
    --lossfn BCE \
    --optimizer SGD \
    --max-epoch 20 \
    --outer-epochs 1\
    --batch-size 128 \
    --lr 0.01\
    --feature-size 2048 \
    --gt list/gt-ucf-R.npy\
    --datasetname UCF \
    --pseudofile ../../minjeong/Unsup_labels/pseudo_labels_swap_90.npy \
    --nalist-path list/nalist_test_i3d.npy \
    --adapter_init_path adapter_init.pt \
    --model_ckpt ../../minjeong/unsupervised_ckpt/UCF_all_cnn_final_20260331_020353_wv5ldb2h.pkl \
    --policy_ckpt safe_meta_policy_lg0_la0/safe_meta_policy_best.pt \

    #--gate_ckpt prefix_gate_best.pt \
    #--gate_threshold 0.5 \
    #--meta_adapter_ckpt meta_adapter_ckpt/meta_adapter_best.pt \
    #--hyper_ckpt prefix_hyper_ckpt/prefix_hyper_best.pt \
    #--conall concat_XD_test.npy\
    #--nalist-path list/nalist_XD_test.npy
    #--xd_feat concat_XD_test.npy\

