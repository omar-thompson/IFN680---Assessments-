import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG, self).__init__()
        
        # TODO: Implement the feature extractor using nn.Sequential
        # Follow the architecture defined in the instructions above.
        self.features = nn.Sequential(
            # --- Block 1 ---
            # (Input: 1 channel -> Output: 64 channels)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 2 ---
            # (Input: 64 channels -> Output: 128 channels)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 3 ---
            # (Input: 128 channels -> Output: 256 channels, note the extra Conv layer here)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_vgg():
    model = VGG(num_classes=2)
    model.load_state_dict(torch.load("vgg.pth"))
    return model