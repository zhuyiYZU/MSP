# import torch
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
#
# # 加载CLIP模型和处理器
# model_name = "./model"  # 预训练的CLIP模型
# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = CLIPProcessor.from_pretrained(model_name)
# model = CLIPModel.from_pretrained(model_name).to(device)
#
# # 定义图像输入
# image_paths = [
#     "F:\\document\\clickbait-pic\\wangyi\\clickbait\\3\\1.jpeg",
#     "F:\\document\\clickbait-pic\\wangyi\\clickbait\\3\\2.jpeg",
#     "F:\\document\\clickbait-pic\\wangyi\\clickbait\\3\\3.jpeg",
# ]
#
# # 定义虚构的文本描述
# text_descriptions = [
#     "网易《饲养手册》刷爆朋友圈，内部公开爆款秘诀！",
# ]
#
# # 编码多张图像和文本
# image_features_list = []
# for image_path, text_description in zip(image_paths, text_descriptions):
#     image = Image.open(image_path)
#     inputs = processor(text=text_description, images=image, return_tensors="pt", padding=True).to(device)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         image_features = outputs.logits_per_image  # 图像特征
#         image_features_list.append(image_features)
#
# # 合并多张图像特征
# combined_image_features = torch.cat(image_features_list, dim=0)  # 在第0维上拼接
#
#
# # 将图像特征与其他文本特征拼接
# text_features = torch.randn(1, 512).to(device)  # 替换成你的其他文本特征
# combined_features = torch.cat((combined_image_features, text_features), dim=-1)
#
# print(combined_features.shape)


import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def encode_images_and_combine_features(image_paths, text_descriptions):
    # 加载CLIP模型和处理器
    model_name = "/home/mjy/anaconda3/envs/pic_prompt/lib/python3.9/site-packages/openprompt/model"  # 预训练的CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # 编码多张图像
    image_features_list = []
    for image_path, text_description in zip(image_paths, text_descriptions):
        image = Image.open(image_path)
        inputs = processor(text=text_description, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.logits_per_image  # 图像特征
            image_features_list.append(image_features)

    # 合并多张图像特征
    combined_image_features = torch.cat(image_features_list, dim=0)  # 在第0维上拼接

    return combined_image_features
# 用法示例
image_paths = [
    "datasets/TextClassification/zongxiang/clickbait/0/1.png",
    "datasets/TextClassification/zongxiang/clickbait/0/2.png",
    "datasets/TextClassification/zongxiang/clickbait/0/3.png",
]

text_descriptions = [
    "网易《饲养手册》刷爆朋友圈，内部公开爆款秘诀！",
]

combined_features = encode_images_and_combine_features(image_paths, text_descriptions)
print(combined_features)