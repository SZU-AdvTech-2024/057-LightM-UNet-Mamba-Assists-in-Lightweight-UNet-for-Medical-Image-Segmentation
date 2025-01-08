import os
import torch
import SimpleITK as sitk


def dice_score(pred_image, gt_image, classes_label=None, smooth=1.0):
    """
    计算n分类数据集的Dice分数。
    :param pred_image: 预测结果张量，形状为 (D, H, W) 或 (N, D, H, W)。
    :param gt_image: 实际标签张量，形状与pred_image相同。
    :param num_classes: 类别总数。
    :param smooth: 用于平滑的常数，防止除以零。
    :return: 每个类别的Dice分数以及平均Dice分数。
    """
    if classes_label is None:
        classes_label = []

    # 如果pred_image是四维的，移除批次维度
    if pred_image.dim() == 4:
        pred_image = pred_image.squeeze(0)
    if gt_image.dim() == 4:
        gt_image = gt_image.squeeze(0)

    dice = torch.zeros(len(classes_label), device=pred_image.device)
    for i in range(len(classes_label)):
        pred = pred_image == classes_label[i]
        true = gt_image == classes_label[i]
        intersection = (pred & true).sum()
        dice[i] = (2. * intersection + smooth) / (pred.sum() + true.sum() + smooth)

    mean_dice = dice.mean()
    return dice, mean_dice

# 计算文件夹所有文件Dice分数

# 预测文件和标注目录
pred_dir = r"E:\Projects\Pycharm\LightM-UNet\data\predict_data\Dataset002_LiverCT_3"
gt_dir = r"E:\Projects\Pycharm\LightM-UNet\data\raw_data\Dataset002_LiverCT\labelsTs"

# 获取所有文件名
pred_files = set(os.listdir(pred_dir))
gt_files = set(os.listdir(gt_dir))

# 找出两个集合中相同的文件名
common_files = pred_files.intersection(gt_files)
print(f"***计算的文件对总数: {len(common_files)}***")

# 计算总的dice便于计算均值
dice_scores_sum = 0
# 计算每个文件对
for pred_file, gt_file in ((os.path.join(pred_dir, file), os.path.join(gt_dir, file)) for file in common_files):
    # 读取预测图像文件
    pred_image = sitk.ReadImage(pred_file)
    # 获取图像数据
    numpy_pred_image = sitk.GetArrayFromImage(pred_image)
    pred = torch.Tensor(numpy_pred_image)

    # 读取真实标签图像文件
    gt_image = sitk.ReadImage(gt_file)
    # 获取图像数据
    numpy_gt_image = sitk.GetArrayFromImage(gt_image)
    gt = torch.Tensor(numpy_gt_image)

    # 计算
    dice_scores, mean_dice_score = dice_score(pred, gt, classes_label=[1, 2, 3], smooth=1)
    dice_scores_sum += mean_dice_score
    print(f"文件: {os.path.basename(pred_file)}")
    print(f"每个标签类别Dice分数: {dice_scores}")
    print(f"平均Dice分数: {mean_dice_score}\n")

print(f"***所有{len(common_files)}个文件平均Dice分数: {dice_scores_sum / len(common_files)}***")


# # 计算单个dice分数
#
# # 读取.nii.gz文件
# pred_image = sitk.ReadImage(r"C:\Users\admin\OneDrive\Desktop\predict_data\LUNG_015.nii.gz")
# # 获取图像数据
# numpy_pred_image = sitk.GetArrayFromImage(pred_image)
# pred = torch.Tensor(numpy_pred_image)
#
# # 读取.nii.gz文件
# gt_image = sitk.ReadImage(r"E:\Datasets\Lung79\Data\labels\LUNG_015.nii.gz")
# # 获取图像数据
# numpy_gt_image = sitk.GetArrayFromImage(gt_image)
# gt = torch.Tensor(numpy_gt_image)
#
# # 计算
# dice_scores, mean_dice_score = dice_score(pred, gt, classes_label=[1, 2, 3], smooth=1e-6)
# print(f"每个标签类别Dice分数: {dice_scores}")
# print(f"平均Dice分数: {mean_dice_score}")
