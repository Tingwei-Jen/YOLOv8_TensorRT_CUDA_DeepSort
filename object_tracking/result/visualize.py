import cv2
import numpy as np
import os
from collections import defaultdict
import argparse

class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

COLORS = np.random.uniform(0, 255, size=(10000, 3))

def draw_bounding_box(img, class_id, track_id, confidence, tlx, tly, width, height):
    label = f"{class_names[class_id]}: {track_id}"
    color = COLORS[track_id]
    cv2.rectangle(img, (tlx, tly), (tlx+width, tly+height), color, 2)

    # Get the width and height of the text box
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_label = tly - 10 if tly - 10 > 10 else tly + 10

    # Draw filled rectangle for text background
    cv2.rectangle(img, (tlx, y_label - text_height - baseline), (tlx + text_width, y_label + baseline), color, thickness=cv2.FILLED)

    # Put the text on top of the rectangle
    cv2.putText(img, label, (tlx, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def generate_img_det(tracker_result_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    data_dict = defaultdict(list)

    with open(tracker_result_path, 'r') as file:
        # 跳過文件的第一行
        next(file)
        # 讀取文件的每一行
        for line in file:
            # 處理每一行的內容
            values = line.strip().split(',')
            # 提取鍵和對應的值
            imgPath = values[0]
            data = values[1:]
            data_dict[imgPath].append(data)

    for imgPath, data in data_dict.items():
        # read image
        image = cv2.imread(imgPath)

        for value in data:
            class_id = int(value[0])
            track_id = int(value[1])
            confidence = float(value[2])
            top_left_x = int(float(value[3]))
            top_left_y = int(float(value[4]))
            width = int(float(value[5]))
            height = int(float(value[6]))
            draw_bounding_box(image, class_id, track_id, confidence, top_left_x, top_left_y, width, height)

        result_path = output_folder_path + '/' + os.path.basename(imgPath)
        cv2.imwrite(result_path, image)

def parse_args():
    parser = argparse.ArgumentParser(description="visualizer for object tracking")
    parser.add_argument('--tracker_result_path', type=str, default='tracker_result.txt', help='path to the tracker result file')
    parser.add_argument('--output_folder_path', type=str, default='img_det', help='path to the output folder')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    generate_img_det(args.tracker_result_path, args.output_folder_path)