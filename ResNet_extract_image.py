import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

def extract_and_merge_features(image_paths):
    # 加载预训练的ResNet模型
    resnet_model = models.resnet50(pretrained=True)

    # 移除最后一层全连接层，保留特征提取器部分
    resnet_feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])

    # 设置模型为评估模式，不进行梯度计算
    resnet_feature_extractor.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 初始化一个空的特征列表
    features_list = []

    for image_path in image_paths:
        # 读取图像并进行预处理
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)  # 添加批次维度

        # 提取图像特征
        with torch.no_grad():
            features = resnet_feature_extractor(image)
            print(features)
            features_list.append(features)
    # 合并多个图像的特征
    merged_features = torch.cat(features_list, dim=0)

    return merged_features

# 示例用法：
image_paths = ['pic_data/tencent/clickbait/1/0.jpeg', 'pic_data/tencent/clickbait/1/3.jpeg','pic_data/tencent/clickbait/1/4.jpeg']  # 替换为你的图像文件路径列表
merged_features = extract_and_merge_features(image_paths)

# 输出合并后的特征向量形状
print("合并的特征向量形状:", merged_features.shape)
