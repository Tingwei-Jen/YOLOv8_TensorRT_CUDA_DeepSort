# YOLOv8_CUDA_DeepSort

YOLOv8 is used for object detection, while CUDA is utilized for image preprocessing and postprocessing after the detection inference. All model inference is performed using TensorRT and implemented in a C++ environment.

## Build Docker container
Dockerfile.yolo includes
- ultralytics
- pytorch
- tensorflow
- onnxrt
- tensorrt
- opencv
- eigen

```bash
sudo docker build -t opencv:opencv --no-cache --rm --file Dockerfile.opencv .
sudo docker build -t detection:yolo --no-cache --rm --file Dockerfile.yolo .
```

## Docker run

```bash
cd YOLOv8_CUDA_DeepSort
sudo docker run -it --rm --gpus all -p 5000:5000 \
--volume="$PWD:/workspace" \
--volume="/home/tingwei/Project/dataset:/workspace/dataset" \
detection:yolo
```