# ============================================================
# FILE: train_karyotype.py
# CHỨC NĂNG: Huấn luyện Swin-T phân loại 24 lớp NST
# PHIÊN BẢN: v2.0 — Tối ưu toàn diện
# ============================================================
import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import argparse
from collections import Counter

from config import settings
from models.swin_karyotype import SwinKaryotype
import torchvision.transforms.functional as TF


# ============================================================
# CUSTOM TRANSFORMS
# ============================================================

class PadToSquare:
    """
    Đệm thêm viền đen (pixel=0) vào ảnh để tạo thành hình vuông
    trước khi Resize, giúp NST không bị bóp méo tỷ lệ.
    
    Ví dụ: Ảnh 32x106 → đệm thành 106x106 → rồi mới Resize về 224x224.
    """
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return TF.pad(img, (p_left, p_top, p_right, p_bottom), self.fill, 'constant')


# ============================================================
# FIX LỖI NGHIÊM TRỌNG: Transform Sharing Bug
# ============================================================
# Khi dùng random_split(), cả train_subset và val_subset đều trỏ về
# CÙNG MỘT dataset gốc. Nếu đổi transform của dataset gốc thì cả
# train và val đều bị ảnh hưởng → training chạy KHÔNG CÓ augmentation.
#
# Giải pháp: Tạo wrapper class riêng cho mỗi split.
# ============================================================

class TransformDataset:
    """
    Wrapper bọc quanh Subset, cho phép gán transform riêng.
    Tránh lỗi chia sẻ transform giữa train và val khi dùng random_split.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        # Lấy đường dẫn ảnh gốc (chưa qua transform nào)
        img_path, label = self.subset.dataset.samples[self.subset.indices[index]]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')  # Ép grayscale → RGB (3 kênh)
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.subset)


# ============================================================
# DATASET
# ============================================================

class KaryotypeDataset(ImageFolder):
    """
    Kế thừa ImageFolder nhưng ép thứ tự class (class_to_idx)
    phải tuân theo đúng settings.KARYOTYPE_LABELS.
    Bằng cách này, Output Index 0 sẽ luôn là "1", Index 22 là "X", Index 23 là "Y".
    """
    def find_classes(self, directory):
        classes = settings.KARYOTYPE_LABELS
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# ============================================================
# CLASS WEIGHTS (Xử lý mất cân bằng dữ liệu)
# ============================================================
# Dataset hiện tại: Class Y chỉ có 45 ảnh, trong khi các class khác
# có 238 ảnh. Nếu không bù trọng số, mô hình sẽ bỏ qua class Y.
# ============================================================

def compute_class_weights(dataset):
    """
    Tính trọng số nghịch đảo theo tần suất cho từng class.
    Class ít mẫu → trọng số cao → Loss đóng góp lớn hơn.
    """
    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    total = len(labels)
    num_classes = len(counter)
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)
        w = total / (num_classes * count)
        weights.append(w)
    weight_tensor = torch.FloatTensor(weights)
    print(f"⚖️  Class weights (xử lý mất cân bằng):")
    for i, w in enumerate(weights):
        label = settings.KARYOTYPE_LABELS[i]
        count = counter.get(i, 0)
        print(f"   Class {label:>2s}: {count:4d} ảnh | weight = {w:.3f}")
    return weight_tensor


# ============================================================
# DATALOADER
# ============================================================

def get_dataloaders():
    img_size = settings.KARYOTYPE_IMG_SIZE

    # Transform cho TRAIN: có Data Augmentation
    transform_train = transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC  # BICUBIC mượt hơn cho ảnh nhỏ
        ),
        # --- Data Augmentation ---
        transforms.RandomHorizontalFlip(p=0.5),
        # KHÔNG dùng RandomVerticalFlip → bảo toàn cấu trúc p-arm / q-arm
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),  # Dịch chuyển nhẹ
            scale=(0.9, 1.1),        # Thu phóng nhẹ
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),  # Che ngẫu nhiên 1 phần nhỏ
    ])

    # Transform cho VAL: KHÔNG có augmentation
    transform_val = transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"📂 Đang tải dữ liệu từ {settings.KARYOTYPE_DATASET_DIR}...")
    
    # Load dataset KHÔNG transform (transform=None) để tránh bug chia sẻ transform
    full_dataset = KaryotypeDataset(root=settings.KARYOTYPE_DATASET_DIR, transform=None)
    
    # Tính class weights TRƯỚC khi chia
    class_weights = compute_class_weights(full_dataset)
    
    # Chia Train / Val (80% / 20%)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(settings.SPLIT_SEED)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Bọc mỗi subset bằng TransformDataset RIÊNG BIỆT → mỗi split có transform độc lập
    train_dataset = TransformDataset(train_subset, transform_train)
    val_dataset = TransformDataset(val_subset, transform_val)

    # num_workers=0 trên Windows để tránh lỗi multiprocessing
    num_workers = 0 if sys.platform == 'win32' else 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=settings.KARYOTYPE_TRAIN_BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=settings.KARYOTYPE_TRAIN_BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"✅ Tổng số ảnh: {total_size} (Train: {train_size}, Val: {val_size})")
    print(f"✅ Số lượng classes: {len(full_dataset.classes)}")
    
    return train_loader, val_loader, class_weights


# ============================================================
# WARMUP + COSINE ANNEALING SCHEDULER
# ============================================================
# Transformer cần warmup để tránh gradient bùng nổ ở đầu training.
# Sau warmup, dùng Cosine Annealing giảm LR mượt mà về 0.
# ============================================================

class WarmupCosineScheduler:
    """Learning rate: warmup tuyến tính → giảm cosine mượt mà."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup tuyến tính: LR tăng dần từ 0 → base_lr
            alpha = (epoch + 1) / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * alpha
        else:
            # Cosine Annealing: LR giảm mượt mà từ base_lr → min_lr
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ============================================================
# HÀM TRAIN CHÍNH
# ============================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Thiết bị huấn luyện: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- Load Data ---
    train_loader, val_loader, class_weights = get_dataloaders()
    class_weights = class_weights.to(device)

    # --- Khởi tạo mô hình ---
    model = SwinKaryotype(num_classes=settings.NUM_KARYOTYPE_CLASSES, dropout=0.3).to(device)
    
    # --- Hàm Loss có Label Smoothing + Class Weights ---
    # Label Smoothing (0.1): giúp mô hình không quá tự tin vào 1 class,
    # tăng khả năng tổng quát hóa (generalization).
    # Class Weights: bù đắp cho class Y (ít mẫu).
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # --- Đường dẫn lưu model ---
    os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
    save_path = settings.KARYOTYPE_CLASSIFIER_WEIGHTS
    
    total_epochs = settings.KARYOTYPE_TRAIN_EPOCHS
    phase1_epochs = min(15, total_epochs // 5)  # Phase 1: ~15 epoch
    best_acc = 0.0
    patience_counter = 0
    patience_limit = 25  # Early stopping nếu 25 epoch liên tiếp không cải thiện

    # ================================================================
    # PHASE 1: Đóng băng backbone — chỉ train lớp Head (15 epoch)
    # ================================================================
    # Mục đích: Cho head (lớp cuối) học nhanh cách phân loại 24 class
    # trước khi tinh chỉnh toàn bộ mô hình.
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Đóng băng Backbone — Train Head ({phase1_epochs} epochs)")
    print(f"{'='*60}")
    
    model.freeze_backbone()
    
    # Chỉ tối ưu các tham số có requires_grad=True (= head)
    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3,           # LR cao vì chỉ train 1 layer nhỏ
        weight_decay=1e-4
    )
    
    for epoch in range(1, phase1_epochs + 1):
        print(f"\n--- Phase 1 | Epoch {epoch}/{phase1_epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p1, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Kỷ lục mới! Val Acc = {best_acc:.2f}% → Đã lưu model")

    # ================================================================
    # PHASE 2: Mở băng toàn bộ — Fine-tune tất cả các layer
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Fine-tune toàn bộ ({total_epochs - phase1_epochs} epochs)")
    print(f"{'='*60}")
    
    model.unfreeze_backbone()
    
    phase2_epochs = total_epochs - phase1_epochs
    
    # LR nhỏ hơn nhiều cho backbone (tránh phá hỏng features đã học từ ImageNet)
    optimizer_p2 = optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if 'head' not in n], 'lr': 5e-5},   # Backbone: LR rất nhỏ
            {'params': [p for n, p in model.named_parameters() if 'head' in n], 'lr': 5e-4},        # Head: LR lớn hơn
        ],
        weight_decay=1e-4
    )
    
    # Warmup 3 epoch + Cosine Annealing
    scheduler_p2 = WarmupCosineScheduler(optimizer_p2, warmup_epochs=3, total_epochs=phase2_epochs)
    
    patience_counter = 0
    
    for epoch in range(1, phase2_epochs + 1):
        scheduler_p2.step(epoch - 1)
        current_lrs = scheduler_p2.get_lr()
        
        print(f"\n--- Phase 2 | Epoch {epoch}/{phase2_epochs} | LR: backbone={current_lrs[0]:.2e}, head={current_lrs[1]:.2e} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p2, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Kỷ lục mới! Val Acc = {best_acc:.2f}% → Đã lưu model")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n⏹️  Early Stopping: {patience_limit} epoch liên tiếp không cải thiện.")
                break
    
    print(f"\n{'='*60}")
    print(f"  ✅ HOÀN TẤT! Best Val Accuracy: {best_acc:.2f}%")
    print(f"  📦 Model đã lưu tại: {save_path}")
    print(f"{'='*60}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Huấn luyện 1 epoch với gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Huấn luyện", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient Clipping: ngăn gradient bùng nổ (rất quan trọng cho Transformer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return total_loss / total, 100. * correct / total


def validate(model, loader, criterion, device):
    """Đánh giá trên tập Validation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Đánh giá", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / total, 100. * correct / total


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Swin-T phân loại 24 lớp NST")
    parser.add_argument("--epochs", type=int, help="Tổng số epoch (ghi đè settings)")
    args = parser.parse_args()
    
    if args.epochs:
        settings.KARYOTYPE_TRAIN_EPOCHS = args.epochs
        
    train()
