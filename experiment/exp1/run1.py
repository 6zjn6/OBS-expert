import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch
from torchvision import transforms
import os
import numpy as np
from timm import create_model
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
 


def get_model():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
    return model




def get_prototype_classifier():
    # 设置随机种子确保可重现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


    project_root = Path(__file__).resolve().parents[1]
    data_path = str(project_root / 'data' / 'organized_radicals')

    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return None, None, None

    # 加载数据集
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    # 按类别分割数据，根据样本数动态调整比例
    train_indices = []
    test_indices = []
    
    # 数据分割统计
    for class_idx in range(len(full_dataset.classes)):
        class_name = full_dataset.classes[class_idx]
        # 找到该类别的所有样本索引
        class_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == class_idx]
        class_count = len(class_indices)
        
        # 根据样本数动态调整分割比例
        if class_count >= 10:
            # 样本数充足，按7:3分割
            train_idx, test_idx = train_test_split(
                class_indices, 
                test_size=0.3,  # 测试集占30%，训练集占70%
                random_state=42,
                stratify=None
            )
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
        elif class_count >= 5:
            # 样本数中等，按8:2分割，确保测试集至少有1个样本
            train_idx, test_idx = train_test_split(
                class_indices, 
                test_size=0.2,  # 测试集占20%，训练集占80%
                random_state=42,
                stratify=None
            )
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
        elif class_count >= 3:
            # 样本数较少，按9:1分割，确保测试集至少有1个样本
            train_idx, test_idx = train_test_split(
                class_indices, 
                test_size=0.1,  # 测试集占10%，训练集占90%
                random_state=42,
                stratify=None
            )
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
        else:
            # 样本数太少，使用全部样本
            train_indices.extend(class_indices)
            test_indices.extend(class_indices)
    
    # 创建训练集和测试集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"\n总数据分割结果:")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  测试集: {len(test_dataset)} 个样本")

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
    for class_idx in range(len(full_dataset.classes)):
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

    # ----------------------------------------------------------
    # 欧氏距离版本 - 计算Top-1、Top-3和Top-5准确率
    predictions = []
    top3_predictions = []
    top5_predictions = []
    
    for feat in test_features:
        dists = {}
        for class_idx, proto in class_prototypes.items():
            dist = np.linalg.norm(feat - proto)
            dists[class_idx] = dist
        
        # Top-1预测
        pred = min(dists, key=dists.get)
        predictions.append(pred)
        
        # Top-3预测
        sorted_dists = sorted(dists.items(), key=lambda x: x[1])
        top3_preds = [item[0] for item in sorted_dists[:3]]
        top3_predictions.append(top3_preds)
        
        # Top-5预测
        top5_preds = [item[0] for item in sorted_dists[:5]]
        top5_predictions.append(top5_preds)
    
    predictions = np.array(predictions)
    top3_predictions = np.array(top3_predictions)
    top5_predictions = np.array(top5_predictions)
    
    # ----------------------------------------------------------
    # 余弦相似度版本
    # predictions_cos = []
    # for feat in test_features:
    #     sims = {}
    #     for class_idx, proto in class_prototypes.items():
    #         sims[class_idx] = cosine_similarity(feat.reshape(1, -1), proto.reshape(1, -1))[0][0]
    #     pred = max(sims, key=sims.get)
    #     predictions_cos.append(pred)
    # predictions_cos = np.array(predictions_cos)
    # ----------------------------------------------------------

    # 计算Top-1准确率
    top1_accuracy = np.mean(predictions == test_labels)
    print(f"\n[欧氏距离] Top-1准确率: {top1_accuracy:.4f} "
          f"({np.sum(predictions == test_labels)}/{len(test_labels)})")
    
    # 计算Top-3准确率
    top3_correct = 0
    for i, (top3_preds, true_label) in enumerate(zip(top3_predictions, test_labels)):
        if true_label in top3_preds:
            top3_correct += 1
    
    top3_accuracy = top3_correct / len(test_labels)
    print(f"[欧氏距离] Top-3准确率: {top3_accuracy:.4f} "
          f"({top3_correct}/{len(test_labels)})")
    
    # 计算Top-5准确率
    top5_correct = 0
    for i, (top5_preds, true_label) in enumerate(zip(top5_predictions, test_labels)):
        if true_label in top5_preds:
            top5_correct += 1
    
    top5_accuracy = top5_correct / len(test_labels)
    print(f"[欧氏距离] Top-5准确率: {top5_accuracy:.4f} "
          f"({top5_correct}/{len(test_labels)})")

    # Top-3正确样本
    print("\nTop-3正确识别的样本:")
    top3_correct_count = 0
    for i, (top3_preds, true) in enumerate(zip(top3_predictions, test_labels)):
        if true in top3_preds:
            top3_correct_count += 1
            pred_names = [full_dataset.classes[pred] for pred in top3_preds]
            print(f"样本 {i}: 真实={full_dataset.classes[true]}, Top-3预测={pred_names}")
    
    # Top-5正确样本
    print("\nTop-5正确识别的样本:")
    top5_correct_count = 0
    for i, (top5_preds, true) in enumerate(zip(top5_predictions, test_labels)):
        if true in top5_preds:
            top5_correct_count += 1
            pred_names = [full_dataset.classes[pred] for pred in top5_preds]
            print(f"样本 {i}: 真实={full_dataset.classes[true]}, Top-5预测={pred_names}")
    
    print(f"\nTop-3正确样本数量: {top3_correct_count}/{len(test_labels)}")
    print(f"Top-5正确样本数量: {top5_correct_count}/{len(test_labels)}")

    # 每类Top-1准确率
    print("\n各类别Top-1准确率:")
    class_correct, class_total = {}, {}
    for pred, true in zip(predictions, test_labels):
        name = full_dataset.classes[true]
        class_total[name] = class_total.get(name, 0) + 1
        if pred == true:
            class_correct[name] = class_correct.get(name, 0) + 1
    for name in sorted(class_total.keys()):
        correct = class_correct.get(name, 0)
        total = class_total[name]
        print(f"  {name}: {correct/total:.4f} ({correct}/{total})")
    
    # 每类Top-3准确率
    print("\n各类别Top-3准确率:")
    class_top3_correct = {}
    for top3_preds, true in zip(top3_predictions, test_labels):
        name = full_dataset.classes[true]
        if true in top3_preds:
            class_top3_correct[name] = class_top3_correct.get(name, 0) + 1
    for name in sorted(class_total.keys()):
        correct = class_top3_correct.get(name, 0)
        total = class_total[name]
        print(f"  {name}: {correct/total:.4f} ({correct}/{total})")
    
    # 每类Top-5准确率
    print("\n各类别Top-5准确率:")
    class_top5_correct = {}
    for top5_preds, true in zip(top5_predictions, test_labels):
        name = full_dataset.classes[true]
        if true in top5_preds:
            class_top5_correct[name] = class_top5_correct.get(name, 0) + 1
    for name in sorted(class_total.keys()):
        correct = class_top5_correct.get(name, 0)
        total = class_total[name]
        print(f"  {name}: {correct/total:.4f} ({correct}/{total})")

    return model, class_prototypes, full_dataset.classes


if __name__ == "__main__":
    get_prototype_classifier()