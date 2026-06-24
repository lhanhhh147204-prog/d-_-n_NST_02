import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
from torchvision import transforms
from medical_pipeline.pipeline.buoc3_0_phan_loai_cum import DualBranchModel
from pathlib import Path
from tqdm import tqdm

def get_transforms():
    """
    Augmentation và Transform chuẩn khớp với lúc Inference.
    Ảnh đầu vào sẽ được chuyển sang 1 kênh xám (Grayscale).
    """
    train_transform = Compose([
        Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = Compose([
        Resize((128, 128)),
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform

def train():
    print("========================================")
    print(" BẮT ĐẦU HUẤN LUYỆN DUAL-BRANCH MODEL")
    print(" (ResNet50 FPN V2 + Swin Transformer)")
    print("========================================")
    
    dataset_path = Path("data_storage/dataset/ccinet_train")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại: {dataset_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Dataset
    print("\n[1] Đang tải dữ liệu...")
    full_dataset = ImageFolder(root=str(dataset_path))
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_transform, val_transform = get_transforms()
    train_dataset.dataset.transform = train_transform
    # Trick để val_dataset dùng đúng val_transform
    val_dataset_copy = ImageFolder(root=str(dataset_path), transform=val_transform)
    val_dataset.dataset = val_dataset_copy
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f" - Train set: {len(train_dataset)} ảnh")
    print(f" - Val set  : {len(val_dataset)} ảnh")
    print(f" - Số class : {len(full_dataset.classes)} -> {full_dataset.classes}")
    
    # 2. Khởi tạo Model
    print("\n[2] Khởi tạo mô hình DualBranchModel...")
    model = DualBranchModel(num_classes=4).to(device)
    
    # Hàm Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Thư mục lưu weight
    save_dir = Path("ai_training/weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_weight_path = save_dir / "improved_model_swin_rn50fpnv2.pth"
    
    num_epochs = 20
    best_acc = 0.0
    
    # 3. Vòng lặp Train
    print("\n[3] Tiến hành huấn luyện...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Train
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100. * correct / total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        print(f" -> Kết quả: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            print(f"    ✨ Cập nhật mô hình tốt nhất! Lưu vào: {best_weight_path.name}")
            torch.save(model.state_dict(), str(best_weight_path))

    print(f"\n✅ HUẤN LUYỆN HOÀN TẤT! Model tốt nhất đạt Val Acc: {best_acc:.2f}%")
    print(f"✅ Vui lòng tải file weights tại: {best_weight_path.absolute()}")

if __name__ == "__main__":
    train()
