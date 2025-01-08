What do experiment planners need to do (these are notes for myself while rewriting nnU-Net, they are provided as is 
without further explanations. These notes also include new features):
- (done) preprocessor name should be configurable via cli
- (done) gpu memory target should be configurable via cli
- (done) plans name should be configurable via cli
- (done) data name should be specified in plans (plans specify the data they want to use, this will allow us to manually 
  edit plans files without having to copy the data folders)
- plans must contain:
    - (done) transpose forward/backward
    - (done) preprocessor name (can differ for each config)
    - (done) spacing
    - (done) normalization scheme
    - (done) target spacing
    - (done) conv and pool op kernel sizes
    - (done) base num features for architecture
    - (done) data identifier
    - num conv per stage?
    - (done) use mask for norm
    - [NO. Handled by LabelManager & dataset.json] num segmentation outputs
    - [NO. Handled by LabelManager & dataset.json] ignore class
    - [NO. Handled by LabelManager & dataset.json] list of regions or classes
    - [NO. Handled by LabelManager & dataset.json] regions class order, if applicable
    - (done) resampling function to be used
    - (done) the image reader writer class that should be used


dataset.json
mandatory:
- numTraining
- labels (value 'ignore' has special meaning. Cannot have more than one ignore_label)
- modalities
- file_ending

optional
- overwrite_image_reader_writer (if absent, auto)
- regions
- region_class_order

### 实验规划器需要做什么（这些是我重写 nnU-Net 时的笔记，未经进一步解释。这些笔记还包括新功能）：
- （完成）预处理器名称应可通过命令行配置
- （完成）GPU 内存目标应可通过命令行配置
- （完成）计划名称应可通过命令行配置
- （完成）数据名称应在计划中指定（计划指定它们想要使用的数据，这将允许我们手动编辑计划文件而无需复制数据文件夹）
- 计划必须包含：
    - （完成）前向/后向转置
    - （完成）预处理器名称（每个配置可以不同）
    - （完成）间距
    - （完成）归一化方案
    - （完成）目标间距
    - （完成）卷积和池化操作的核大小
    - （完成）架构的基础特征数量
    - （完成）数据标识符
    - 每阶段的卷积数量？
    - （完成）是否使用掩码进行归一化
    - [不。由 LabelManager 和 dataset.json 处理] 分割输出的数量
    - [不。由 LabelManager 和 dataset.json 处理] 忽略的类别
    - [不。由 LabelManager 和 dataset.json 处理] 区域或类别的列表
    - [不。由 LabelManager 和 dataset.json 处理] 区域类别的顺序（如果适用）
    - （完成）要使用的重采样函数
    - （完成）要使用的图像读取器/写入器类

### dataset.json
#### 必填项：
- numTraining
- labels（值 'ignore' 有特殊含义。不能有多个 ignore_label）
- modalities
- file_ending

#### 可选项：
- overwrite_image_reader_writer（如果不存在，则自动）
- regions
- region_class_order