import os
import argparse
import torch
import torchvision
from model import ResNet18Reid
from PIL import Image

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = ResNet18Reid(num_classes=1501, reid=False)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # move model to device
    model.eval()
    
    # image transform
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load image
    image = Image.open(args.img_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加批次維度
    image = image.to(device)  # move image to device

    with torch.no_grad():
       outputs = model(image)
       probabilities = torch.nn.functional.softmax(outputs, dim=1)
       max_prob, predicted = torch.max(probabilities, 1)  # 獲取預測結果
    return predicted.item(), max_prob.item()

def feature_extractor(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = ResNet18Reid(num_classes=1501, reid=True)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # move model to device
    model.eval()

    # image transform
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load image
    image = Image.open(args.img_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加批次維度
    image = image.to(device)  # move image to device

    with torch.no_grad():
       outputs = model(image)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Resnet18Reid predict")
    parser.add_argument("--img_path", default='img.jpg', type=str)
    parser.add_argument('--checkpoint_path', type=str, required=True, help='check_point path, default: None')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    predicted_class, prob = predict(args)
    print('Predicted class:', predicted_class, 'probability:', format(prob, '.5f'))
    feature = feature_extractor(args)
    print('Feature:', feature.shape)
    print('Feature:', feature)