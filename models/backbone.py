import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model, load_checkpoint

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Efficientb7_timm(nn.Module):
    def __init__(self, num_classes=8):
        super(Efficientb7_timm, self).__init__()
        model = create_model('tf_efficientnet_b7', pretrained=False,num_classes = 8)
        ckpt_path = 'output/20230203-023625-tf_efficientnet_b7_ns/Best_f1_score.pth.tar'
        load_checkpoint(model, ckpt_path)

        self.base = nn.Sequential(*list(model.children())[:-1])

        self.classifier = model.classifier
        print(self.classifier)
    def forward(self, image):
        feature_map = self.base(image) # (16,2048,7,7)
        feature = F.normalize(feature_map, dim=1) # (16,512)
        out = self.classifier(feature) # (16,8)

        return out