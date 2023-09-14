# Inroduction
This repository contains a library to use [YOLOv7](https://github.com/WongKinYiu/yolov7) object detection mode in C++. It also contains a demo program to run YOLOv7 on images or video streams (like webcams).

# Build
The library needs `ONNX Runtime` as a dependency. You can find pre-compiled packages or build from source [here](https://github.com/microsoft/onnxruntime). The demo program needs `OpenCV` which you can install using your package manager or other sources. If you want to run the model on CUDA, then you will need to link to CUDA's library directory. If you want to build the demo program, you can follow these steps:

``` bash
git clone git@github.com:ShahriarRezghi/yolov7-cxx.git
cd yolov7-cxx
mkdir build
cd build
cmake \
    -DONNX_RUNTIME_PATH=<path/to/onnx/runtime/root> \
    -DCUDA_LIBRARIES_PATH=<path/to/cuda/toolkit/root> \
    ..
cmake --build .
```

This will give you an executable file named `yolov7_demo` that you can run to perform object detection:

``` bash
# For image detection:
./yolov7_demo -m <path/to/model.onnx> -i <path/to/image.jpg>
# Or for video stream detection:
./yolov7_demo -m <path/to/model.onnx> -v </path/to/video/device> --cuda 0
```

You must pass a video device like /dev/video0 when trying to detect from video stream. If you want to use the project as a library, you can add it as a `CMake` subdirectory like this:

``` cmake
add_subdirectory(yolov7_cxx)
set(ONNX_RUNTIME_PATH "<path/to/onnx/runtime/root>")
set(CUDA_LIBRARIES_PATH "<path/to/cuda/toolkit/root>")
add_executable(MyTarget main.cpp)
target_link_libraries(MyTarget PUBLIC yolov7_cxx)
```

# Minimal Example
A simple code is provided below to showcase the interface of the library. Also, the API of the interface is commented, and more details can be found there.

``` c++
#include <yolov7_cxx/yolov7_cxx.h>

int main()
{
    YOLOv7_CXX::YOLOv7 yolo(model_path, cuda_device);
    int image_size = yolo.image_size();
    float *data = /* pointer to continuous NCHW images data */;
    YOLOv7_CXX::Shape shape = {1, 3, image_size, image_size};
    auto results = yolo.detect(data, shape);
}
```

# Contributing
You can report bugs, ask questions, and request features on [issues page](../../issues).

# License
This library is licensed under BSD 3-Clause permissive license. You can read it [here](LICENSE).
