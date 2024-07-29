# YOLOv8_CUDA_DeepSort

YOLOv8 is used for object detection, while CUDA is utilized for image preprocessing and postprocessing after the detection inference. All model inference is performed using TensorRT and implemented in a C++ environment.

## Docker
### Build Docker container
Dockerfile.yolo includes
- ultralytics
- pytorch
- tensorflow
- onnxrt
- tensorrt
- opencv
- eigen

To build the Docker containers, use the following commands:

```bash
sudo docker build -t opencv:opencv --no-cache --rm --file Dockerfile.opencv .
sudo docker build -t detection:yolo --no-cache --rm --file Dockerfile.yolo .
```

## Run Docker Container
To run the Docker container, navigate to the project directory and use the following command:

```bash
cd YOLOv8_CUDA_DeepSort
sudo docker run -it --rm --gpus all -p 5000:5000 \
--volume="$PWD:/workspace" \
--volume="/home/tingwei/Project/dataset:/workspace/dataset" \
detection:yolo
```

## Generate Yolov8 onnx model and convert to tensorRT
First, generate the YOLOv8 ONNX model by running the following commands:
```bash
cd yolov8
python export.py
```

## Generate extractor onnx model and training for it.

Follow the steps in the [README.md](reid/README.md) of 'reid' folder to generate and train the extractor ONNX model.

## Convert ONNX model to TensortRT engine.

Follow the steps in the [README.md](convert_onnx_to_engine_cpp/README.md) of the convert_onnx_to_engine_cpp folder to convert the ONNX model to a TensorRT engine.

## Object Tracking

### Usage
1. **Build the project** 

    Create a build directory and compile the code using CMake and Make:

```bash
cd object_tracking
mkdir build
cd build
cmake ..
make
```
2. **Run the Object Tracking Module**

Use the following command to run the object tracking module, replacing [detector_engine_path] with the path to your detector engine model, [extractor_engine_path] with the path to your extractor engine model, [image_folder_path] with the path to your testing images, and [tracker_result_output_path] with the location for the tracking result text file.

```bash
./object_tracking [detector_engine_path] [extractor_engine_path] [image_folder_path] [tracker_result_output_path]
```

3. **Visulize the result**

```python
usage: visualize.py [--tracker_result_path]
                    [--output_folder_path]
```