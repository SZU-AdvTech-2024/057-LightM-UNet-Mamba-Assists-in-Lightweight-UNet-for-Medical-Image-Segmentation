# '''
# 增加文件文件名通道
# '''
# import os
#
# def add_suffix_before_extension(directory, suffix="_0000"):
#     # 检查目录是否存在
#     if not os.path.exists(directory):
#         print("指定的目录不存在，请检查路径。")
#         return
#
#     # 遍历目录下的所有文件
#     for filename in os.listdir(directory):
#         # 获取文件的完整路径
#         old_file = os.path.join(directory, filename)
#         # 检查是否是文件
#         if os.path.isfile(old_file):
#             # 分离文件名和扩展名
#             name, ext = os.path.splitext(filename)
#             # 创建新的文件名（在文件名和扩展名之间插入后缀）
#             new_filename = f"{name}{suffix}{ext}"
#             # 获取新文件的完整路径
#             new_file = os.path.join(directory, new_filename)
#             # 重命名文件
#             os.rename(old_file, new_file)
#             print(f"文件 {filename} 已重命名为 {new_filename}")
#
# # 使用示例
# # 请将下面的路径替换为你想要处理的目录路径
# directory_path = "/data/ZhipengLiu/projects/LightM-UNet/data/raw_data/Dataset002_LiverCT/labelsTs"
# add_suffix_before_extension(directory_path)

# '''
# 填充序号为4为数字
# '''
# import os
# import re
#
# def format_filename_with_zero_padding(directory):
#     # 检查目录是否存在
#     if not os.path.exists(directory):
#         print("指定的目录不存在，请检查路径。")
#         return
#
#     # 遍历目录下的所有文件
#     for filename in os.listdir(directory):
#         # 获取文件的完整路径
#         old_file = os.path.join(directory, filename)
#         # 检查是否是文件
#         if os.path.isfile(old_file):
#             # 使用正则表达式匹配文件名中的序号部分
#             match = re.search(r'(\w+)_(\d+)_(\d+)', filename)
#             if match:
#                 # 提取文件名部分、序号部分和后缀部分
#                 prefix, number, suffix = match.groups()
#                 # 将序号部分格式化为四位数字
#                 formatted_number = f"{int(number):04d}"
#                 # 创建新的文件名
#                 new_filename = f"{prefix}_{formatted_number}_{suffix}" + ".nii"
#                 # 获取新文件的完整路径
#                 new_file = os.path.join(directory, new_filename)
#                 # 重命名文件
#                 os.rename(old_file, new_file)
#                 print(f"文件 {filename} 已重命名为 {new_filename}")
#
# # 使用示例
# # 请将下面的路径替换为你想要处理的目录路径
# directory_path = "/data/ZhipengLiu/projects/LightM-UNet/data/raw_data/Dataset002_LiverCT/labelsTs"
# format_filename_with_zero_padding(directory_path)
#
# '''
# 检测训练文件和标签是否有差异
# '''
# import os
#
# def compare_directories(dir1, dir2):
#     # 获取两个目录中的文件名集合
#     files_in_dir1 = set(os.listdir(dir1))
#     files_in_dir2 = set(os.listdir(dir2))
#
#     # 找出只在第一个目录中的文件
#     unique_to_dir1 = files_in_dir1 - files_in_dir2
#     # 找出只在第二个目录中的文件
#     unique_to_dir2 = files_in_dir2 - files_in_dir1
#
#     # 找出在两个目录中都存在的文件
#     common_files = files_in_dir1 & files_in_dir2
#
#     print("在目录1中独有的文件：")
#     for file in unique_to_dir1:
#         print(file)
#
#     print("\n在目录2中独有的文件：")
#     for file in unique_to_dir2:
#         print(file)
#
#     print("\n两个目录中都有的文件：")
#     for file in common_files:
#         print(file)
#
# # 使用示例
# # 请将下面的路径替换为你想要比较的目录路径
# directory_path1 = "/data/ZhipengLiu/projects/LightM-UNet/data/raw_data/Dataset002_LiverCT/imagesTr"
# directory_path2 = "/data/ZhipengLiu/projects/LightM-UNet/data/raw_data/Dataset002_LiverCT/labelsTr"
#
# compare_directories(directory_path1, directory_path2)

# '''
# 读取文件
# '''
# import SimpleITK as sitk
# import numpy as np
#
# # 读取.nii.gz文件
# image = sitk.ReadImage('E:\Files\labelliver_0000_0000.nii.gz', sitk.sitkFloat32)
#
# # 将图像转换为数组
# image_array = sitk.GetArrayFromImage(image)
#
# # 现在你可以使用image_array来分析或处理图像数据
# fl = image_array.flatten()
# print(np.unique(fl))

import os
import re

def remove_specific_suffix(directory, suffix="_0000"):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以.nii.gz结尾
        if filename.endswith('.nii.gz'):
            # 构建完整的文件路径
            old_file = os.path.join(directory, filename)
            # 使用正则表达式替换后缀
            new_filename = re.sub(suffix + r'\.nii\.gz$', '.nii.gz', filename)
            # 如果新旧文件名不同，则重命名
            if new_filename != filename:
                new_file = os.path.join(directory, new_filename)
                os.rename(old_file, new_file)
                print(f"Renamed '{filename}' to '{new_filename}'")

# 使用示例
# 请将下面的路径替换为你想要处理的目录路径
directory_path = "/data/ZhipengLiu/projects/LightM-UNet/data/raw_data/Dataset002_LiverCT/labelsTs"
remove_specific_suffix(directory_path)