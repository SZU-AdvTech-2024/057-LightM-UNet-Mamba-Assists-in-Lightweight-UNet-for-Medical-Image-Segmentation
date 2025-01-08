from copy import deepcopy
import numpy as np

"""文件主要用于计算神经网络的池化层和卷积层的参数，以确保网络的结构符合特定的要求"""
def get_shape_must_be_divisible_by(net_numpool_per_axis):
    """
    计算每个轴上的池化层数量，并返回一个数组，表示每个轴上的形状必须被多少整除
    """
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisible by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """
    this is the same as get_pool_and_conv_props_v2 from old nnunet;根据给定的间距、补丁大小、最小特征图大小和最大池化层数量，计算池化层和卷积层的参数

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :param max_numpool:
    :return:
    """
    # todo review this code
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = [[1] * len(spacing)]
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2*min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]

        # find axis that are within factor of 2 within smallest spacing
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]

        # max_numpool constraint
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break

        # now we need to find kernel sizes
        # kernel sizes are initialized to 1. They are successively set to 3 when their associated axis becomes within
        # factor 2 of min_spacing. Once they are 3 they remain 3
        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            else:
                if current_spacing[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))
        #print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by


# 池化层和卷积层的参数计算：这些函数帮助确定网络中池化层和卷积层的核大小，确保网络的结构合理且符合特定的约束条件。
# 形状调整：通过 pad_shape 函数，确保输入数据的形状能够被网络的池化层整除，避免出现不匹配的问题。
# 网络设计辅助：这些函数为网络设计提供了重要的参数支持，使得网络能够在不同的数据集和任务上表现良好