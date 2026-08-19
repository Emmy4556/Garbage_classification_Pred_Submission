import torch
import torch.nn as nn


class GarbageCNN(nn.Module):
    """From-scratch CNN for garbage classification (no pretrained backbone)."""

    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)

        self.dropout2d = nn.Dropout2d(0.2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout2d(x)
        x = self.block2(x)
        x = self.dropout2d(x)
        x = self.block3(x)
        x = self.dropout2d(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return self.classifier(x)


def load_model(checkpoint_path, device="cpu"):
    """Loads checkpoint dict {model_state_dict, class_names, img_size, mean, std}."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = GarbageCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
