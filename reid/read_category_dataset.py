import os
import random
import argparse
import json
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 使用 cv2 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        image = Image.fromarray(image)  # 转换为 PIL 图像

        if self.transform:
            image = self.transform(image)

        return image, label


# valid_rate is the ratio of validation set to the total dataset
def read_and_split_data(data_root, valid_rate=0.2):
    assert os.path.exists(data_root), 'dataset root: {} does not exist.'.format(data_root)
    class_names = [cls for cls in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, cls))]
    class_names.sort()
    class_indices = {name: i for i, name in enumerate(class_names)}
    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)
    
    train_images_path = []
    train_labels = []
    val_images_path = []
    val_labels = []
    per_class_num = []

    supported = ['.jpg', '.JPG', '.png', '.PNG']
    for cls in class_names:
        cls_path = os.path.join(data_root, cls)
        images_path = [os.path.join(cls_path, i) for i in os.listdir(cls_path)
                       if os.path.splitext(i)[-1] in supported]
    
        images_label = class_indices[cls] 
        per_class_num.append(len(images_path))
        val_path = random.sample(images_path, int(len(images_path) * valid_rate))

        for img_path in images_path:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_labels.append(images_label)
            else:
                train_images_path.append(img_path)
                train_labels.append(images_label)

    assert len(train_images_path) > 0, "number of training images must greater than zero"
    assert len(val_images_path) > 0, "number of validation images must greater than zero"

    plot_distribution = False
    if plot_distribution:
        plt.bar(range(len(class_names)), per_class_num, align='center')
        plt.xticks(range(len(class_names)), class_names)

        for i, v in enumerate(per_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('classes')
        plt.ylabel('numbers')
        plt.title('the distribution of dataset')
        plt.show()
        plt.savefig('distribution.png')

    return [train_images_path, train_labels], [val_images_path, val_labels], len(class_names)

def parse_args():
    parser = argparse.ArgumentParser(description="dataset prepare")
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument('--valid_rate', type=float, default=0.2)
    return parser.parse_args()

# main
if __name__ == "__main__":
    args = parse_args()
    read_and_split_data(args.data_dir, args.valid_rate)
