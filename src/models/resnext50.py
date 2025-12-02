import torch
import torch.nn as nn
import logging
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# RESNEXT BOTTLENECK BLOCK
# ============================================================================

class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block with grouped convolutions

    Architecture:
      1x1 conv: expand to intermediate channels
      3x3 grouped conv: groups=32, 4 channels per group (cardinality=32, width=4)
      1x1 conv: project to output channels
      Shortcut path: identity or 1x1 conv for dimension matching
      BatchNorm + ReLU after each conv layer
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=32, width_per_group=4):
        super(ResNeXtBottleneck, self).__init__()

        # Calculate intermediate channels
        width = int(out_channels / 4 * width_per_group / 64 * groups)

        # 1x1 conv reduce/expand to intermediate channels
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3 grouped conv
        self.conv2 = nn.Conv2d(
            width, width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)

        # 1x1 conv: project to output channels
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 conv to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut path
        identity = self.shortcut(identity)

        # add residual connection
        out += identity
        out = self.relu(out)

        return out


# ============================================================================
# RESNEXT-50 MODEL
# ============================================================================

class ResNeXt50(nn.Module):
    """
    ResNeXt-50 (32x4d)

    Architecture ->
      Stem: 7x7 conv (stride=2)  64 channels, BN, ReLU, 3x3 MaxPool (stride=2)
      Layer1: 3 bottleneck blocks (64 256 channels, stride=1)
      Layer2: 4 bottleneck blocks (256 512 channels, stride=2 on first)
      Layer3: 6 bottleneck blocks (512 1024 channels, stride=2 on first)
      Layer4: 3 bottleneck blocks (1024 2048 channels, stride=2 on first)
      Head: AdaptiveAvgPool2d  Flatten  Linear(2048, num_classes)
    """

    def __init__(self, num_classes=50, groups=32, width_per_group=4):
        super(ResNeXt50, self).__init__()

        self.groups = groups
        self.width_per_group = width_per_group
        self.num_classes = num_classes

        # ========== STEM ==========
        # 7x7 
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ========== RESIDUAL LAYERS ==========
        # Layer 1
        self.layer1 = self._make_layer(
            in_channels=64,
            out_channels=256,
            num_blocks=3,
            stride=1
        )

        # Layer 2
        self.layer2 = self._make_layer(
            in_channels=256,
            out_channels=512,
            num_blocks=4,
            stride=2
        )

        # Layer 3 
        self.layer3 = self._make_layer(
            in_channels=512,
            out_channels=1024,
            num_blocks=6,
            stride=2
        )

        # Layer 4
        self.layer4 = self._make_layer(
            in_channels=1024,
            out_channels=2048,
            num_blocks=3,
            stride=2
        )

        # ========== CLASSIFICATION HEAD ==========
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

        logger.info(f"ResNeXt-50 initialized with {num_classes} classes")

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        layers.append(
            ResNeXtBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group
            )
        )

        # Remaining blocks with stride=1
        for _ in range(1, num_blocks):
            layers.append(
                ResNeXtBottleneck(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    groups=self.groups,
                    width_per_group=self.width_per_group
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # initialize model weights using Kaiming 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem: (B, 3, 224, 224) -> (B, 64, 56, 56)
        x = self.stem(x)

        # Layer 1 (B, 64, 56, 56) -> (B, 256, 56, 56)
        x = self.layer1(x)

        # Layer 2 (B, 256, 56, 56) -> (B, 512, 28, 28)
        x = self.layer2(x)

        # Layer 3 (B, 512, 28, 28) -> (B, 1024, 14, 14)
        x = self.layer3(x)

        # Layer 4 (B, 1024, 14, 14) -> (B, 2048, 7, 7)
        x = self.layer4(x)

        # Classification head: (B, 2048, 7, 7) -> (B, num_classes)
        x = self.avgpool(x)  # (B, 2048, 1, 1)
        x = self.flatten(x)  # (B, 2048)
        x = self.fc(x)       # (B, num_classes)

        return x

    # ========== TRANSFER LEARNING UTILITIES ==========

    def freeze_for_phase1(self):
        """
        Phase 1: Layer4 + FC training

        Frozen: Stem + Layer1 + Layer2 + Layer3
        Trainable: Layer4 + FC
        """
        # Freeze early layers
        self.stem.eval()
        for param in self.stem.parameters():
            param.requires_grad = False

        self.layer1.eval()
        for param in self.layer1.parameters():
            param.requires_grad = False

        self.layer2.eval()
        for param in self.layer2.parameters():
            param.requires_grad = False

        self.layer3.eval()
        for param in self.layer3.parameters():
            param.requires_grad = False

        # Unfreeze Layer4
        self.layer4.train()
        for param in self.layer4.parameters():
            param.requires_grad = True

        # Unfreeze FC layer
        for param in self.fc.parameters():
            param.requires_grad = True

        logger.info("Phase 1 freeze: Layer4 + FC trainable")
    
    def freeze_for_phase2(self):
        """
        Phase 2: Layer3-4 + FC training

        Frozen: Stem + Layer1 + Layer2
        Trainable: Layer3 + Layer4 + FC
        """
        # Freeze early layers
        self.stem.eval()
        for param in self.stem.parameters():
            param.requires_grad = False

        self.layer1.eval()
        for param in self.layer1.parameters():
            param.requires_grad = False

        self.layer2.eval()
        for param in self.layer2.parameters():
            param.requires_grad = False

        # Unfreeze Layer3
        self.layer3.train()
        for param in self.layer3.parameters():
            param.requires_grad = True

        # Unfreeze Layer4
        self.layer4.train()
        for param in self.layer4.parameters():
            param.requires_grad = True

        # Unfreeze FC layer
        for param in self.fc.parameters():
            param.requires_grad = True

        logger.info("Phase 2 freeze: Layer3-4 + FC trainable")

    def freeze_for_phase3(self):
        """
        Phase 3: Layer2-4 + FC training

        Frozen: Stem + Layer1
        Trainable: Layer2 + Layer3 + Layer4 + FC
        """
        # Freeze early layers
        self.stem.eval()
        for param in self.stem.parameters():
            param.requires_grad = False

        self.layer1.eval()
        for param in self.layer1.parameters():
            param.requires_grad = False

        # Unfreeze Layer2
        self.layer2.train()
        for param in self.layer2.parameters():
            param.requires_grad = True

        # Unfreeze Layer3
        self.layer3.train()
        for param in self.layer3.parameters():
            param.requires_grad = True

        # Unfreeze Layer4
        self.layer4.train()
        for param in self.layer4.parameters():
            param.requires_grad = True

        # Unfreeze FC layer
        for param in self.fc.parameters():
            param.requires_grad = True

        logger.info("Phase 3 freeze: Layer2-4 + FC trainable")

    def freeze_for_phase4(self):
        """
        Phase 4 ALL layers trainable

        Frozen: None
        Trainable: Stem + Layer1 + Layer2 + Layer3 + Layer4 + FC
        """
        # Unfreeze all layers
        self.stem.train()
        for param in self.stem.parameters():
            param.requires_grad = True

        self.layer1.train()
        for param in self.layer1.parameters():
            param.requires_grad = True

        self.layer2.train()
        for param in self.layer2.parameters():
            param.requires_grad = True

        self.layer3.train()
        for param in self.layer3.parameters():
            param.requires_grad = True

        self.layer4.train()
        for param in self.layer4.parameters():
            param.requires_grad = True

        # FC layer
        for param in self.fc.parameters():
            param.requires_grad = True

        logger.info("Phase 4 freeze: ALL layers trainable")

    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True

        logger.info("All layers unfrozen for fine-tuning")

    def load_pretrained_imagenet(self, reinit_fc=True):
        logger.info("Loading pretrained ResNeXt-50 (32x4d) weights from ImageNet...")

        # Load pretrained model from torchvision
        pretrained_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()

        # get current model state dict
        model_dict = self.state_dict()

        # filter out FC layer weights
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and not k.startswith('fc.')
        }

        # update current model with pretrained weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        logger.info(f"Loaded {len(pretrained_dict)} pretrained layers")

        # reinitialize FC layer for new task
        if reinit_fc:
            nn.init.normal_(self.fc.weight, 0, 0.01)
            nn.init.constant_(self.fc.bias, 0)
            logger.info(f"Reinitialized FC layer for {self.num_classes} classes")

        return self

    def get_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        return trainable_params, total_params

    def print_architecture_summary(self):
        """Print a summary of the model architecture"""
        trainable, total = self.get_trainable_params()

        logger.info("=" * 60)
        logger.info("ResNeXt-50 Architecture Summary")
        logger.info("=" * 60)
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Groups (cardinality): {self.groups}")
        logger.info(f"Width per group: {self.width_per_group}")
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"Frozen parameters: {total - trainable:,}")
        logger.info("=" * 60)


# ============================================================================
# MODEL FACTORY FUNCTION
# ============================================================================

def create_resnext50(num_classes=50, pretrained=True, freeze_backbone=False):
    model = ResNeXt50(num_classes=num_classes)

    if pretrained:
        model.load_pretrained_imagenet(reinit_fc=True)

    if freeze_backbone:
        model.freeze_backbone()

    model.print_architecture_summary()

    return model


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    # test the model
    logger.info("Testing ResNeXt-50 model...")

    model = create_resnext50(num_classes=50, pretrained=False, freeze_backbone=False)

    dummy_input = torch.randn(4, 3, 224, 224)

    # forward pass
    output = model(dummy_input)

    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {output.shape}")

    # test freeze && unfreeze
    model.freeze_backbone()
    trainable, total = model.get_trainable_params()
    logger.info(f"After freezing backbone: {trainable:,} / {total:,} trainable")

    model.unfreeze_all()
    trainable, total = model.get_trainable_params()
    logger.info(f"After unfreezing all: {trainable:,} / {total:,} trainable")

    logger.info("ResNeXt-50 model test completed successfully!")
