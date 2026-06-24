# ==========================================
# BƯỚC 3.0: PHÂN LOẠI CỤM NHIỄM SẮC THỂ
# CHỨC NĂNG: Phân loại cụm thành 0 (Đơn), 1 (Chạm), 2 (Chồng), 3 (Chồng Chạm)
# ==========================================

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import warnings
from pathlib import Path

# Cấu trúc Dual Branch (ResNet50 + Swin-T)
class DualBranchModel(nn.Module):
    def __init__(self, num_classes):
        super(DualBranchModel, self).__init__()
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        detection_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.resnet_fpn = detection_model.backbone

        new_conv1 = nn.Conv2d(1, self.resnet_fpn.body.conv1.out_channels,
                              kernel_size=self.resnet_fpn.body.conv1.kernel_size,
                              stride=self.resnet_fpn.body.conv1.stride,
                              padding=self.resnet_fpn.body.conv1.padding,
                              bias=self.resnet_fpn.body.conv1.bias is not None)
        with torch.no_grad():
            new_conv1.weight.copy_(self.resnet_fpn.body.conv1.weight.data.mean(dim=1, keepdim=True))
        self.resnet_fpn.body.conv1 = new_conv1
        self.resnet_fpn_feature_dim = 256

        self.swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        # torchvision Swin Transformer có lớp Conv2d đầu tiên tại features[0][0]
        try:
            old_conv = self.swin.features[0][0]
            new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding,
                                 bias=old_conv.bias is not None)
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.data.mean(dim=1, keepdim=True))
            self.swin.features[0][0] = new_conv
        except Exception as e:
            print("Cảnh báo: Không thể thay đổi Conv1 của Swin Transformer:", e)

        self.swin_feature_dim = self.swin.head.in_features
        self.swin.head = nn.Identity()

        fusion_dim = self.resnet_fpn_feature_dim + self.swin_feature_dim
        self.att_fc = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=1))
        self.fusion_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(fusion_dim, num_classes))

    def forward(self, x):
        features = self.resnet_fpn(x)
        feat_map = list(features.values())[-1]
        resnet_feat = nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
        resnet_feat = resnet_feat.view(resnet_feat.size(0), -1)

        swin_feat = self.swin(x)

        fused = torch.cat((resnet_feat, swin_feat), dim=1)
        att_weights = self.att_fc(fused)
        res_weight = att_weights[:, 0].unsqueeze(1)
        swin_weight = att_weights[:, 1].unsqueeze(1)
        fused_weighted = torch.cat((resnet_feat * res_weight, swin_feat * swin_weight), dim=1)

        out = self.fusion_fc(fused_weighted)
        return out


class ClusterClassifier:
    def __init__(self, model_path: Path, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_classes = 4 # 0: Đơn, 1: Chạm, 2: Chồng, 3: Chồng Chạm
        
        self.has_weights = model_path.exists()
        if not self.has_weights:
            warnings.warn(f"⚠️ [FALLBACK] Không tìm thấy {model_path}. Hệ thống sẽ giả định tất cả là NST Đơn (Class 0).")
            self.model = None
        else:
            self.model = DualBranchModel(num_classes=self.num_classes).to(self.device)
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                warnings.warn(f"⚠️ Lỗi load weights CCINet: {e}. Fallback -> Class 0.")
                self.has_weights = False

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, img_bgr) -> int:
        """
        Dự đoán lớp của cụm NST.
        Input: cv2 image (BGR hoặc Grayscale)
        Output: int (0, 1, 2, 3)
        """
        if not self.has_weights:
            # Fallback an toàn: Coi như cụm đơn giản
            return 0
            
        if not isinstance(img_bgr, Image.Image):
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            # Chuyển OpenCV BGR sang RGB cho PIL
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        else:
            img_pil = img_bgr

        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = output.argmax(dim=1).item()
            
        return prediction
