import os
import shutil
import random
import argparse
import json

def prepare_market1501(source_dir, target_dir):
    assert os.path.exists(source_dir), 'source_dir: {} does not exist.'.format(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f'Directory {target_dir} created.')

    supported = ['.jpg', '.JPG', '.png', '.PNG']


    # 瀏覽來源資料夾中的所有文件
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            # 提取 ID
            file_id = filename.split('_')[0]
            # 定義目標資料夾路徑
            file_target_dir = os.path.join(target_dir, file_id)
            # 確保 ID 資料夾存在
            os.makedirs(file_target_dir, exist_ok=True)
            # 定義源文件和目標文件的完整路徑
            src_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(file_target_dir, filename)
            # 複製文件
            shutil.copy2(src_file, dest_file)

def parse_args():
    parser = argparse.ArgumentParser(description="prepare market1501 dataset")
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prepare_market1501(args.source_dir, args.target_dir)
