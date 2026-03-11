import torch.nn as nn
from torchvision.models import efficientnet_b0


class MultiTaskEfficientNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = efficientnet_b0(pretrained=True)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.4)

        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)
        self.race_head = nn.Linear(in_features, 5)

    def forward(self, x):

        features = self.backbone(x)

        # Apply dropout
        features = self.dropout(features)

        age = self.age_head(features)
        gender = self.gender_head(features)
        race = self.race_head(features)

        return age, gender, race