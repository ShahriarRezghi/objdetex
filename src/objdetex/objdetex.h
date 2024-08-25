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

#include <objdetex/config.h>

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/// @brief The ObjDetEx namespace contains classes, functions, and type definitions
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

/// @brief Represents the size or index, typically used for dimensions or class indices.
using Size = int64_t;

/// @brief Represents a standard string type used for names, labels, or other text-based data.
using String = std::string;

/// @brief Template alias for a vector.
///
/// This alias is used to create a vector of any specified type.
///
/// @tparam T The type of elements stored in the vector.
template <typename T>
using Vector = std::vector<T>;

/// @brief Template alias for a shared pointer.
///
/// This alias is used for managing shared ownership of dynamically allocated objects.
///
/// @tparam T The type of the object managed by the shared pointer.
template <typename T>
using Pointer = std::shared_ptr<T>;

/// @brief Holds information about a detected object in an image.
struct Detection
{
    /// @brief The x-coordinate of the top-left corner of the bounding box.
    double x;

    /// @brief The y-coordinate of the top-left corner of the bounding box.
    double y;

    /// @brief The width of the bounding box.
    double w;

    /// @brief The height of the bounding box.
    double h;

    /// @brief The index of the detected object, typically corresponding to the class index in the model.
    Size index;

    /// @brief The class name of the detected object.
    String name;

    /// @brief The confidence level of the detection.
    double confidence;
};

/// @brief Overloads the output stream operator for the `Detection` struct.
///
/// This operator allows a `Detection` object to be printed to an output stream in a human-readable format.
///
/// @param os The output stream.
/// @param detection The `Detection` object to be printed.
/// @return The output stream with the `Detection` information.
std::ostream &operator<<(std::ostream &os, const Detection &detection);

/// @brief A type alias representing the shape of a tensor.
///
/// This is a vector of `Size` elements, typically used to define the dimensions
/// of a tensor in terms of width, height, depth, etc.
using Shape = Vector<Size>;

/// @brief A type alias representing a collection of detections.
///
/// This is a vector of `Detection` objects, each representing an individual detection result.
using Detections = Vector<Detection>;

/// @brief A class representing a tensor used in the object detection process.
///
/// The `Tensor` class is a flexible container for holding data of various types
/// (such as `float` or `int64_t`) and allows for custom memory management through
/// the use of deleter functions.
struct Tensor
{
    /// @brief Enumeration of supported tensor data types.
    ///
    /// This enum defines the types of data that a `Tensor` can hold, such as
    /// 32-bit floating point (`Float32`) or 64-bit integer (`Int64`).
    enum Type
    {
        Float32,  ///< 32-bit floating point type.
        Int64,    ///< 64-bit integer type.
    };

    /// @brief A template alias for a custom deleter function.
    ///
    /// This alias defines a function that can be used to properly delete
    /// dynamically allocated tensor data when custom memory management is needed.
    ///
    /// @tparam T The type of the data to be deleted.
    template <typename T>
    using Deleter = std::function<void(T *)>;

    /// @brief The type of the tensor data (Float32 or Int64).
    Type type{Type::Float32};

    /// @brief The shape of the tensor, representing the dimensions of the data.
    Shape shape;

    /// @brief A shared pointer to the tensor data.
    Pointer<void> data;

    /// @brief Default constructor for the `Tensor` class.
    ///
    /// Initializes an empty `Tensor` object with no data, type, or shape defined.
    Tensor() = default;

    /// @brief Constructs a `Tensor` for `float` data without ownership.
    ///
    /// This constructor creates a `Tensor` that holds a pointer to `float` data without
    /// taking ownership, meaning the data will not be deleted when the `Tensor` is destroyed.
    ///
    /// @param data Pointer to the `float` data.
    /// @param shape The shape of the tensor.
    Tensor(float *data, const Shape &shape);

    /// @brief Constructs a `Tensor` for `int64_t` data without ownership.
    ///
    /// This constructor creates a `Tensor` that holds a pointer to `int64_t` data without
    /// taking ownership, meaning the data will not be deleted when the `Tensor` is destroyed.
    ///
    /// @param data Pointer to the `int64_t` data.
    /// @param shape The shape of the tensor.
    Tensor(int64_t *data, const Shape &shape);

    /// @brief Constructs a `Tensor` for `float` data with a custom deleter.
    ///
    /// This constructor creates a `Tensor` that takes ownership of the `float` data,
    /// meaning the data will be deleted using the provided deleter function when the `Tensor` is destroyed.
    ///
    /// @param data Pointer to the `float` data.
    /// @param shape The shape of the tensor.
    /// @param deleter A custom deleter function to be used for deleting the data.
    Tensor(float *data, const Shape &shape, Deleter<float> deleter);

    /// @brief Constructs a `Tensor` for `int64_t` data with a custom deleter.
    ///
    /// This constructor creates a `Tensor` that takes ownership of the `int64_t` data,
    /// meaning the data will be deleted using the provided deleter function when the `Tensor` is destroyed.
    ///
    /// @param data Pointer to the `int64_t` data.
    /// @param shape The shape of the tensor.
    /// @param deleter A custom deleter function to be used for deleting the data.
    Tensor(int64_t *data, const Shape &shape, Deleter<int64_t> deleter);

    /// @brief Boolean conversion operator.
    ///
    /// This operator allows a `Tensor` object to be evaluated in a boolean context,
    /// such as in an `if` statement. It returns `true` if the `data` pointer is not `nullptr`,
    /// indicating that the tensor contains data.
    ///
    /// @return `true` if the tensor contains data, `false` otherwise.
    inline operator bool() const { return bool(data); }
};

/// Internal structure, not to be used publicly
struct Impl;

/// @brief A class for performing object detection using various models.
///
/// The `Detector` class provides an interface to load different object detection models, such as YOLO and RT-DETR,
/// and run inference on input images. It supports multiple model types and allows for specifying a device for
/// running the inference, such as a CPU or CUDA-enabled GPU.
struct Detector
{
    /// @brief Enumeration of the supported object detection models.
    ///
    /// This enum defines the different types of object detection models that the `Detector` class can handle.
    enum Type
    {
        YOLOv7,    ///< The YOLOv7 object detection model.
        YOLOv8,    ///< The YOLOv8 object detection model.
        YOLOv9,    ///< The YOLOv9 object detection model.
        YOLOv10,   ///< The YOLOv10 object detection model.
        YOLO_NAS,  ///< The YOLO-NAS object detection model.
        RT_DETR,   ///< The RT-DETR object detection model.
    };

    /// @brief Constructs a `Detector` object with a specified model type.
    ///
    /// This constructor initializes the `Detector` by loading the specified model from the given path.
    /// An optional device ID can be provided to specify the computation device (e.g., CPU or GPU).
    ///
    /// @param type The type of the object detection model to be used.
    /// @param path The file path to the model.
    /// @param device The device ID to use for inference (-1 for CPU, device ID for GPU).
    Detector(Type type, const String &path, Size device = -1);

    /// @brief Performs object detection on the input images with a default detection confidence threshold.
    ///
    /// This operator allows the `Detector` object to be used as a function to detect objects in input tensor images
    /// with a specified confidence threshold. It returns a vector of detections for each image.
    ///
    /// The input `images` tensor should have the shape `BxCxHxW` or `CxHxW` where `B` is the batch size,
    /// `C` is the number of channels, `H` is the height, and `W` is the width. The data type of the tensor
    /// should be `float32`.
    ///
    /// @param images A `Tensor` representing the input images with the shape `BxCxHxW` or `CxHxW` and float32 data
    /// type.
    /// @param threshold The detection confidence threshold, a value between 0 and 1 (default is 0.6).
    /// @return Vector<Detections> A vector of detection results for each input image.
    inline Vector<Detections> operator()(Tensor images, double threshold = .6) const
    {
        return (*this)(images, {}, threshold);
    }

    /// @brief Performs object detection on the input images with specified dimensions and detection confidence
    /// threshold.
    ///
    /// This operator allows the `Detector` object to be used as a function to detect objects in input tensor images
    /// with specific dimensions and a confidence threshold. It returns a vector of detections for each image.
    ///
    /// The input `images` tensor should have the shape `BxCxHxW` or `CxHxW` where `B` is the batch size,
    /// `C` is the number of channels, `H` is the height, and `W` is the width. The data type of the tensor
    /// should be `float32`.
    ///
    /// The `dimensions` tensor should have the shape `Bx2`, where `B` is the batch size, and each pair of values
    /// represents the original width and height of the corresponding image before resizing. The data type of the tensor
    /// should be `int64`
    ///
    /// @param images A `Tensor` representing the input images with the shape `BxCxHxW` or `CxHxW` and float32 data
    /// type.
    /// @param dimensions A `Tensor` representing the original dimensions of the input images with the shape `Bx2`.
    /// @param threshold The detection confidence threshold, a value between 0 and 1 (default is 0.6).
    /// @return Vector<Detections> A vector of detection results for each input image.
    Vector<Detections> operator()(Tensor images, Tensor dimensions, double threshold = .6) const;

    /// @brief Gets the expected input image size for the specified model.
    ///
    /// This function returns the size (width and height) of the input image that the model expects.
    ///
    /// @return Size The size (width and height) of the input image expected by the model.
    Size imageSize() const;

private:
    /// @brief The type of the object detection model.
    Type type;

    /// @brief Pointer to the internal implementation (`Impl`) of the `Detector`.
    Pointer<Impl> impl;
};
}  // namespace ObjDetEx
