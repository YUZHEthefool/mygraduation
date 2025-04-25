import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torchvision.models import Inception_V3_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# 1. 设置环境与加载模型
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义数据预处理
val_test_transforms = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载测试集
test_data = datasets.ImageFolder('data/test', transform=val_test_transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
class_names = test_data.classes
num_classes = len(class_names)

# 初始化模型
model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
model.load_state_dict(torch.load('E:/model/inception_v3_animal_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# 2. 模型评估
# -------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test Accuracy: {accuracy:.2f}%")

# -------------------------------
# 3. 分类报告
# -------------------------------
report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)

# -------------------------------
# 4. 绘制混淆矩阵
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# -------------------------------
# 5. 训练过程可视化（使用真实训练数据）
# -------------------------------
# 从训练日志中提取的真实数据
train_loss = [2.0067, 1.7809, 1.7759, 1.7580, 1.7632,
              1.7539, 1.7683, 1.7691, 1.7638, 1.7648]  # 训练损失
val_loss = [0.1728, 0.1506, 0.1490, 0.1391, 0.1243,
            0.1288, 0.1254, 0.1260, 0.1249, 0.1213]     # 验证损失
val_acc = [95.53, 96.01, 95.76, 96.03, 96.57,
           96.32, 96.41, 96.41, 96.41, 96.35]          # 验证准确率
epochs = range(1, 11)  # 共10个epoch

# Loss 曲线（训练损失 + 验证损失）
plt.figure()
plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)  # 显示所有epoch刻度
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# Accuracy 曲线（仅验证准确率）
plt.figure()
plt.plot(epochs, val_acc, 'g-o', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.xticks(epochs)  # 显示所有epoch刻度
plt.ylim(95, 97)    # 聚焦精度变化范围
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png')
plt.show()
