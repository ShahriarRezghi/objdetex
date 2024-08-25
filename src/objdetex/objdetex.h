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
    Type type{Type::Float32};

    /// The shape of the tensor, representing the dimensions of the data.
    Shape shape;

    /// A shared pointer to the tensor data.
    Pointer<void> data;

    /// @brief Default constructor for the `Tensor` class.
    ///
    /// This constructor initializes an empty `Tensor` object with no data, type, or shape defined.
    /// The resulting `Tensor` will evaluate to `false` in a boolean context until it is assigned valid data.
    Tensor() = default;

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

    /// @brief Boolean conversion operator.
    ///
    /// This operator allows a `Tensor` object to be evaluated in a boolean context, such as in an `if` statement.
    /// It returns `true` if the `data` pointer is not `nullptr`, indicating that the tensor contains data.
    ///
    /// @return `true` if the tensor contains data, `false` otherwise.
    inline operator bool() const { return bool(data); }
};

/// Internal structure, not to be used publicly
struct Impl;

struct Detector
{
    enum Type
    {
        RT_DETR,
        YOLOv10,
        YOLOv9,
        YOLOv8,
        YOLOv7,
    };

    Detector(Type type, const String &path, Size device = -1);

    Vector<Detections> operator()(Tensor images, double threshold = .6, Tensor dimensions = {}) const;

    Size imageSize() const;

private:
    Type type;
    Pointer<Impl> impl;
};
}  // namespace ObjDetEx
