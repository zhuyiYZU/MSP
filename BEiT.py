from transformers import BeitImageProcessor, BeitForImageClassification
import torch
from PIL import Image

def Beit_extract_image_features(image_paths):
    # 加载 BEiT 模型和图像处理器
    processor = BeitImageProcessor.from_pretrained('./model/beit')
    model = BeitForImageClassification.from_pretrained('./model/beit')

    # 用于存储图像特征的列表
    image_features = []

    for image_path in image_paths:
        # 打开图像文件并创建 PIL 图像对象
        image = Image.open(image_path)

        # 使用图像处理器对图像进行处理
        inputs = processor(images=image, return_tensors="pt")

        # 使用模型进行推理并获取图像特征
        outputs = model(**inputs)

        image_feature = outputs.logits.mean(dim=0)  # 使用均值池化获取编码表示
        # image_feature = outputs.logits
        print(image_feature.shape)
        # 添加图像特征到列表中
        image_features.append(image_feature.unsqueeze(0))

    print(len(image_features))
    # 拼接图像特征
    concatenated_features = torch.cat(image_features, dim=0)

    return concatenated_features

# 使用示例
image_paths = ['./datasets/TextClassification/tencent/clickbait/2/image_0.jpg', './datasets/TextClassification/tencent/clickbait/2/image_1.jpg', './datasets/TextClassification/tencent/clickbait/2/image_2.jpg']
features = Beit_extract_image_features(image_paths)
print("Concatenated Features Shape:", features.shape)
