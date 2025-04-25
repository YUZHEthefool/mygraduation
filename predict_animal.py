import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image

def load_model(model_path, num_classes):
    # 加载预训练的InceptionV3模型
    model = models.inception_v3(weights=None)  # 不加载默认权重
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(image_path, model, class_names, device):
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载和预处理图像
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    # 返回预测的类别名称
    return class_names[class_idx]

if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载训练数据以获取类别名称
    train_data = datasets.ImageFolder('data/train', transform=transforms.ToTensor())
    class_names = train_data.classes

    # 加载模型
    model_path = 'E:/model/inception_v3_animal_classifier.pth'
    num_classes = len(class_names)
    model = load_model(model_path, num_classes).to(device)

    # 输入图像目录路径
    image_dir = 'D:/project/GraduationProject/test/'  # 替换为你的图像目录路径

    # 批量预测
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            predicted_class = predict_image(image_path, model, class_names, device)
            print(f"{image_name}: {predicted_class}")