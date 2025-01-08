## 5则交叉验证
#for fold in {0..4}
#do
#    # echo "nnUNetv2_train 1 3d_lowres $fold"
#    TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0"  nnUNetv2_train 1 3d_fullres $fold -device cuda --npz
#    # if the -device is cuda then we can set CUDA_VISIBLE_DEVICES=num_of_gpus
#done

TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0" nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerLightMUNet -device cuda --npz