/// BSD 3-Clause License
///
/// Copyright (c) 2023-2024, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
/// 1. Redistributions of source code must retain the above copyright notice, this
///    list of conditions and the following disclaimer.
///
/// 2. Redistributions in binary form must reproduce the above copyright notice,
///    this list of conditions and the following disclaimer in the documentation
///    and/or other materials provided with the distribution.
///
/// 3. Neither the name of the copyright holder nor the names of its
///    contributors may be used to endorse or promote products derived from
///    this software without specific prior written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
/// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
/// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
/// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
/// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
/// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
/// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ObjDetEx
{
template <typename T>
void print(T &&value)
{
    std::cout << value << " " << std::endl;
}
template <typename Head, typename... Tail>
void print(Head &&head, Tail &&...tail)
{
    std::cout << head << " ";
    print(std::forward<Tail>(tail)...);
}
inline void print() { std::cout << std::endl; }

/// The size or index type, used for dimensions of input data or output detections.
using Size = int64_t;

/// Standard string type used for names or labels.
using String = std::string;

/// Alias for a standard vector.
template <typename T>
using Vector = std::vector<T>;

/// Alias for a shared pointer.
template <typename T>
using Pointer = std::shared_ptr<T>;

/// Holds information about a detected object in an image.
struct Detection
{
    /// The x-coordinate of the top-left corner of the bounding box.
    double x;

    /// The y-coordinate of the top-left corner of the bounding box.
    double y;

    /// The width of the bounding box.
    double w;

    /// The height of the bounding box.
    double h;

    /// The index of the detected object, typically corresponding to the class index in the model.
    Size index;

    /// The class name of the detected object.
    String name;

    /// The confidence level of the detection.
    double confidence;
};

/// Overload the ostream operator
std::ostream &operator<<(std::ostream &os, const Detection &detection);

/// Shape of an input or output tensor, used to define the dimensions of the data.
using Shape = Vector<Size>;

/// List of detection results, where each detection corresponds to an object found in the input image.
using Detections = Vector<Detection>;

/// Internal structure, not to be used publicly
struct Detector;

/// @brief A wrapper class for handling tensor data used in the object detection process.
///
/// This class provides a flexible way to manage tensor data, including options for custom memory management
/// using custom deleters. The `Tensor` class can hold either float or int64_t data, depending on the specific
/// needs of the application.
struct Tensor
{
    /// Enumeration of the supported tensor data types.
    enum Type
    {
        Float32,  ///< 32-bit floating point type.
        Int64,    ///< 64-bit integer type.
    };

    /// @brief A template alias for a custom deleter function.
    ///
    /// The deleter function is used to manage the memory of the tensor data when custom deletion behavior is needed.
    template <typename T>
    using Deleter = std::function<void(T *)>;

    /// The type of the tensor data (Float32 or Int64).
    Type type;

    /// The shape of the tensor, representing the dimensions of the data.
    Shape shape;

    /// A shared pointer to the tensor data.
    Pointer<void> data;

    /// @brief Constructs a Tensor object to hold a pointer to float data without ownership.
    ///
    /// This constructor does not take ownership of the data, meaning the data will not be deleted when the Tensor is
    /// destroyed.
    ///
    /// @param data Pointer to the float data.
    /// @param shape The shape of the tensor.
    Tensor(float *data, const Shape &shape);

    /// @brief Constructs a Tensor object to hold a pointer to int64_t data without ownership.
    ///
    /// This constructor does not take ownership of the data, meaning the data will not be deleted when the Tensor is
    /// destroyed.
    ///
    /// @param data Pointer to the int64_t data.
    /// @param shape The shape of the tensor.
    Tensor(int64_t *data, const Shape &shape);

    /// @brief Constructs a Tensor object to hold a pointer to float data with a custom deleter.
    ///
    /// This constructor takes ownership of the data, meaning the data will be deleted using the provided deleter
    /// function when the Tensor is destroyed.
    ///
    /// @param data Pointer to the float data.
    /// @param shape The shape of the tensor.
    /// @param deleter A custom deleter function to be used for deleting the data.
    Tensor(float *data, const Shape &shape, Deleter<float> deleter);

    /// @brief Constructs a Tensor object to hold a pointer to int64_t data with a custom deleter.
    ///
    /// This constructor takes ownership of the data, meaning the data will be deleted using the provided deleter
    /// function when the Tensor is destroyed.
    ///
    /// @param data Pointer to the int64_t data.
    /// @param shape The shape of the tensor.
    /// @param deleter A custom deleter function to be used for deleting the data.
    Tensor(int64_t *data, const Shape &shape, Deleter<int64_t> deleter);
};

/// @brief A class for performing object detection using the YOLOv7 model.
///
/// This class provides an interface to load a YOLOv7 ONNX model and run inference on input images.
struct YOLOv7
{
    /// @brief Constructs a YOLOv7 detector.
    ///
    /// This constructor initializes the YOLOv7 model by loading it from the specified path.
    /// An optional CUDA device ID can be specified to run the model on a GPU.
    ///
    /// @param modelPath The file path to the ONNX model.
    /// @param cudaDevice The CUDA device ID to use for inference (-1 for CPU).
    YOLOv7(const String &modelPath, Size cudaDevice = -1);

    /// @brief Performs object detection on the input image.
    ///
    /// This operator allows the YOLOv7 object to be used as a function to detect objects in an input tensor image.
    ///
    /// @param image A Tensor representing the input image.
    /// @return Vector<Detections> A vector of detection results, where each element corresponds to a detected object.
    Vector<Detections> operator()(Tensor image);

    /// Gets the expected input image size for the YOLOv7 model.
    ///
    /// @return Size The size (width and height) of the input image expected by the model.
    Size imageSize() const;

private:
    Pointer<Detector> impl;
};

/// @brief A class for performing object detection using the YOLOv8 model.
///
/// This class provides an interface to load a YOLOv8 ONNX model and run inference on input images.
struct YOLOv8
{
    /// @brief Constructs a YOLOv8 detector.
    ///
    /// This constructor initializes the YOLOv8 model by loading it from the specified path.
    /// An optional CUDA device ID can be specified to run the model on a GPU.
    ///
    /// @param modelPath The file path to the ONNX model.
    /// @param cudaDevice The CUDA device ID to use for inference (-1 for CPU).
    YOLOv8(const String &modelPath, Size cudaDevice = -1);

    /// @brief Performs object detection on the input image.
    ///
    /// This operator allows the YOLOv8 object to be used as a function to detect objects in an input tensor image.
    ///
    /// @param image A Tensor representing the input image.
    /// @param threshold The confidence threshold for filtering detections (default is 0.45).
    /// @return Vector<Detections> A vector of detection results, where each element corresponds to a detected object.
    Vector<Detections> operator()(Tensor image, float threshold = .45);

    /// @brief Gets the expected input image size for the YOLOv8 model.
    ///
    /// @return Size The size (width and height) of the input image expected by the model.
    Size imageSize() const;

private:
    Pointer<Detector> impl;
};

using YOLOv9 = YOLOv8;

/// @brief A class for performing object detection using the YOLOv10 model.
///
/// This class provides an interface to load a YOLOv10 ONNX model and run inference on input images.
struct YOLOv10
{
    /// @brief Constructs a YOLOv10 detector.
    ///
    /// This constructor initializes the YOLOv10 model by loading it from the specified path.
    /// An optional CUDA device ID can be specified to run the model on a GPU.
    ///
    /// @param modelPath The file path to the ONNX model.
    /// @param cudaDevice The CUDA device ID to use for inference (-1 for CPU).
    YOLOv10(const String &modelPath, Size cudaDevice = -1);

    /// @brief Performs object detection on the input image.
    ///
    /// This operator allows the YOLOv10 object to be used as a function to detect objects in an input tensor image.
    ///
    /// @param image A Tensor representing the input image.
    /// @param threshold The confidence threshold for filtering detections (default is 0.6).
    /// @return Vector<Detections> A vector of detection results, where each element corresponds to a detected object.
    Vector<Detections> operator()(Tensor image, double threshold = .6);

    /// @brief Gets the expected input image size for the YOLOv10 model.
    ///
    /// @return Size The size (width and height) of the input image expected by the model.
    Size imageSize() const;

private:
    Pointer<Detector> impl;
};

/// @brief A class for performing object detection using the RT-DETR model.
///
/// This class provides an interface to load an RT-DETR ONNX model and run inference on input images.
struct RT_DETR
{
    /// @brief Constructs an RT_DETR detector.
    ///
    /// This constructor initializes the RT-DETR model by loading it from the specified path.
    /// An optional CUDA device ID can be specified to run the model on a GPU.
    ///
    /// @param modelPath The file path to the ONNX model.
    /// @param cudaDevice The CUDA device ID to use for inference (-1 for CPU).
    RT_DETR(const String &modelPath, Size cudaDevice = -1);

    /// @brief Performs object detection on the input image with a specified threshold.
    ///
    /// This operator allows the RT_DETR object to be used as a function to detect objects in an input tensor image.
    /// A threshold can be set to filter out detections with low confidence.
    ///
    /// @param image A Tensor representing the input image.
    /// @param dims A Tensor representing the dimensions of the image.
    /// @param threshold The confidence threshold for filtering detections (default is 0.6).
    /// @return Vector<Detections> A vector of detection results, where each element corresponds to a detected object.
    Vector<Detections> operator()(Tensor image, Tensor dims, double threshold = .6);

    /// Gets the expected input image size for the RT-DETR model.
    ///
    /// @return Size The size (width and height) of the input image expected by the model.
    Size imageSize() const;

private:
    Pointer<Detector> impl;
};
}  // namespace ObjDetEx
