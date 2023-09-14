/// BSD 3-Clause License
///
/// Copyright (c) 2023, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
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

#include <memory>
#include <string>
#include <vector>

namespace YOLOv7_CXX
{
/// This class represents a detection instance
struct Result
{
    /// Starting x point of the bounding box
    double x;

    /// Starting y point of the bounding box
    double y;

    /// Width of the bounding box
    double w;

    /// Height of the bounding box
    double h;

    /// Detected object class index
    int64_t index;

    /// Detected object class name
    std::string name;

    /// Confidence of object detection
    double confidence;
};

struct Impl;

/// Shape used by the library
using Shape = std::vector<int64_t>;

/// List of detected instances in an image
using Results = std::vector<Result>;

/// The YOLO model class that does the detections
struct YOLOv7
{
    /// Initialize the YOLOv7 class with the given parameters
    /// @param model_path the path to the ONNX model of YOLOv7
    /// @param cuda_device CUDA device index to use, uses CPU if value is -1
    YOLOv7(const std::string &model_path, int64_t cuda_device = -1);

    /// Get input image size to the model
    /// @return input image size of the model
    int64_t image_size() const;

    /// Run the input through the model and return the detections
    /// @param data pointer to the start of image data in the form of continuous CHW or NCHW
    /// @param shape shape of the input image, can be CHW or NCHW
    /// @return list of list of detection instances, the size of top level vector is N
    std::vector<Results> detect(float *data, Shape shape);

private:
    std::shared_ptr<Impl> impl;
};
}  // namespace YOLOv7_CXX
