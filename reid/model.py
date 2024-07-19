import torch
import torchvision

class ResNet18Reid(torch.nn.Module):
    def __init__(self, num_classes=1000, reid=False):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.reid = reid
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.model.fc(x)
        return x

if __name__ == '__main__':
    model = ResNet18Reid(num_classes=50, reid=True)
    model.eval()
    print(model)
    with torch.no_grad():
        x = torch.randn(20, 3, 128, 64)
        y = model(x)
        print("Output shape:", y.shape)