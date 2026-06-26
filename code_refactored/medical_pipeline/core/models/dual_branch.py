import torch
import torch.nn as nn
from torchvision import models

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
