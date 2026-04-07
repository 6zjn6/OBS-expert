import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
import os
import numpy as np
from timm import create_model
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist   # 用于欧氏距离


def get_model():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置离线模式，避免网络下载问题
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    try:
        print("🔧 Loading DinoV2 model...")
        model = create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        print("✅ DinoV2 model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load DinoV2 model: {e}")
        print("🔧 Trying with offline mode...")
        try:
            # 尝试使用本地缓存的模型
            model = create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
            print("⚠️ Loaded model without pretrained weights")
            return model
        except Exception as e2:
            print(f"❌ Failed to load model even in offline mode: {e2}")
            raise e2


def get_prototype_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(current_dir, 'data', 'organized_radicals')
    test_data_path = os.path.join(current_dir, 'data', 'organized_radicals')

    if not os.path.exists(train_data_path):
        print(f"❌ 训练数据路径不存在: {train_data_path}")
        return None, None, None
    if not os.path.exists(test_data_path):
        print(f"❌ 测试数据路径不存在: {test_data_path}")
        return None, None, None

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("训练集类别统计：")
    class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    for k, v in sorted(class_counts.items()):
        print(f"  {k}: {v} 个样本")

    model.eval()
    train_features, train_labels = [], []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            outputs = model(images)
            train_features.append(outputs.cpu().numpy())
            train_labels.append(labels.cpu().numpy())

    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # 标准化
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    train_features = (train_features - mean) / std

    # 构建每个类别的原型（平均向量）
    class_prototypes = {}
    for class_idx in range(len(train_dataset.classes)):
        class_feats = train_features[train_labels == class_idx]
        if len(class_feats) == 0:
            continue
        prototype = np.mean(class_feats, axis=0)
        class_prototypes[class_idx] = prototype

    # 提取测试集特征
    test_features, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            test_features.append(outputs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_features = (test_features - mean) / std

    # （原准确率计算与打印已全部删除）

    return model, class_prototypes, train_dataset.classes


if __name__ == "__main__":
    get_prototype_classifier()