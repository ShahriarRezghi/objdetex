# Inroduction
This repository contains a library to use various object detection models in C++. It also contains a demo program to run the models on images or video streams (like webcams). Supported models are listed below.

+ [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
+ [YOLOv10](https://github.com/THU-MIG/yolov10)
+ [YOLOv9](https://github.com/WongKinYiu/yolov9)
+ [YOLOv8](https://github.com/ultralytics/ultralytics)
+ [YOLOv7](https://github.com/WongKinYiu/yolov7)

# Build
The library needs `ONNX Runtime` as a dependency. You can find pre-compiled packages or build from source [here](https://github.com/microsoft/onnxruntime). The demo program needs `OpenCV` which you can install using your package manager or other sources. If you want to run the model on CUDA, then you will need to link to CUDA's library directory. If you want to build the demo program, you can follow these steps:

``` bash
git clone git@github.com:ShahriarRezghi/objdetex.git
cd objdetex
mkdir build
cd build
cmake \
    -DONNX_RUNTIME_PATH=<path/to/onnx/runtime/root> \
    -DCUDA_LIBRARIES_PATH=<path/to/cuda/toolkit/root> \
    ..
cmake --build .
```

This will give you an executable file named `objdetex_demo` that you can run to perform object detection:

``` bash
# For image detection:
./objdetex_demo -m <path/to/model.onnx> -i <path/to/image.jpg>
# Or for video stream detection:
./objdetex_demo -m <path/to/model.onnx> -v </path/to/video/device> --cuda 0
```

You must pass a video device like /dev/video0 when trying to detect from video stream. If you want to use the project as a library, you can add it as a `CMake` subdirectory like this:

``` cmake
add_subdirectory(objdetex)
set(ONNX_RUNTIME_PATH "<path/to/onnx/runtime/root>")
set(CUDA_LIBRARIES_PATH "<path/to/cuda/toolkit/root>")
add_executable(MyTarget main.cpp)
target_link_libraries(MyTarget PUBLIC objdetex)
```

# Minimal Example
A simple code is provided below to showcase the interface of the library. Also, the API of the library is documented, and more details can be found there.

``` c++
#include <objdetex/objdetex.h>

int main()
{
    using namespace ObjDetEx;
    Detector detector(Detector::RT_DETR, "<path/to/onnx/model>");

    Size batchSize = 1;
    double detectionThreshold = .6;

    // Fill this with batchSizex3x640x640 image data
    float *imagePtr = nullptr;
    
    // Fill this with batchSizex2 dimension data, not needed for YOLO models
    // NOTE: 2 is width and height of the original images before resizing to 640x640
    int64_t *dimensionPtr = nullptr;

    auto detections = detector(Tensor(imagePtr, {batchSize, 3, 640, 640}),  //
                               detectionThreshold, Tensor(dimensionPtr, {batchSize, 2}));

    // Use the detections
    return 0;
}
```

# Contributing
You can report bugs, ask questions, and request features on [issues page](../../issues).

# License
This library is licensed under BSD 3-Clause permissive license. You can read it [here](LICENSE).
