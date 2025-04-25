import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import gradio as gr

def load_model(model_path, num_classes):
    model = models.inception_v3(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(image, model, class_names, device):
    preprocess = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    return class_names[class_idx]

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载训练数据以获取类别名称
train_data = datasets.ImageFolder('data/train', transform=transforms.ToTensor())
class_names = train_data.classes

# 加载模型
model_path = 'inception_v3_animal_classifier.pth'
num_classes = len(class_names)
model = load_model(model_path, num_classes).to(device)

# 定义Gradio接口
def gradio_predict(image):
    return predict_image(image, model, class_names, device)

iface = gr.Interface(fn=gradio_predict,
                     inputs=gr.Image(type="pil"),
                     outputs=gr.Text(),
                     title="Animal Classifier",
                     description="Upload an image of an animal and get the predicted class.")

# 启动Gradio应用
iface.launch()