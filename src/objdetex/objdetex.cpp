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

#include "objdetex.h"

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <codecvt>
#include <locale>
#include <sstream>
#include <stdexcept>

#define OBJDETEX_ASSERT(expr, msg)                                            \
    if (!static_cast<bool>(expr))                                             \
    {                                                                         \
        std::ostringstream stream;                                            \
        std::string file = __FILE__, func = __PRETTY_FUNCTION__;              \
        stream << "Assertion at " << file << ":" << __LINE__ << "->" << func; \
        stream << ":\n\t" << msg << std::endl;                                \
        throw std::runtime_error(stream.str());                               \
    }

namespace ObjDetEx
{
using Sequence = const char *;

const char *classNames[] = {
    "<unknown>",    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",
    "train",        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter",
    "bench",        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",
    "elephant",     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",
    "tie",          "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",
    "cup",          "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",
    "sandwich",     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",
    "cake",         "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",
    "tv",           "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",
    "oven",         "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",
    "scissors",     "teddy bear",     "hair drier", "toothbrush"};

#ifdef _WIN32  // ORTCHAR_T is wchar_t when _WIN32 is defined.
#define TO_ONNX_STR(stdStr) std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().from_bytes(stdStr).c_str()
#else
#define TO_ONNX_STR(stdStr) stdStr.c_str()
#endif

Ort::SessionOptions createOptions(int64_t cuda_device)
{
    Ort::SessionOptions options;
    if (cuda_device >= 0) Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, cuda_device));
    return options;
}

int64_t elements(const std::vector<int64_t> &shape)
{
    int64_t target = 1;
    for (auto item : shape) target *= item;
    return target;
}

void transposeMatrix(const float *input, float *output, int64_t x, int64_t y)
{
    for (int64_t i = 0; i < x; ++i)
        for (int64_t j = 0; j < y; ++j)  //
            output[j * x + i] = input[i * y + j];
}

std::pair<Size, String> getClass(float index)
{
    Size idx = std::nearbyint(index);
    idx = (std::max<Size>)(-1, (std::min<Size>)(idx, 79));
    return {idx, classNames[idx + 1]};
}

Tensor::Tensor(float *data, const Shape &shape) : Tensor(data, shape, [](float *) {}) {}

Tensor::Tensor(int64_t *data, const Shape &shape) : Tensor(data, shape, [](int64_t *) {}) {}

Tensor::Tensor(float *data, const Shape &shape, Deleter<float> deleter)
{
    this->type = Float32;
    this->shape = shape;
    this->data = Pointer<float>(data, deleter);
}

Tensor::Tensor(int64_t *data, const Shape &shape, Deleter<int64_t> deleter)
{
    this->type = Int64;
    this->shape = shape;
    this->data = Pointer<int64_t>(data, deleter);
}

std::ostream &operator<<(std::ostream &os, const Detection &detection)
{
    os << "Detection ["                                                      //
       << "class=" << detection.name                                         //
       << ", confidence=" << std::round(detection.confidence * 10000) / 100  //
       << ", bounding-box=(x=" << detection.x << ", y=" << detection.y       //
       << ", w=" << detection.w << ", h=" << detection.h << ")]";            //
    return os;
}

struct Impl
{
    Ort::Env env;
    Ort::SessionOptions options;
    Ort::Session session;

    Impl(const String &modelPath, int64_t cudaDevice)
        : env(ORT_LOGGING_LEVEL_WARNING, "ObjDetEx"),
          options(createOptions(cudaDevice)),
          session(env, TO_ONNX_STR(modelPath), options)
    {
    }

    Size imageSize() const
    {
        auto info = session.GetInputTypeInfo(0);
        return info.GetTensorTypeAndShapeInfo().GetShape().back();
    }

    Vector<Ort::Value> operator()(const Vector<Tensor> &tensors, const Vector<Sequence> &inputNames,
                                  const Vector<Sequence> &outputNames)
    {
        Vector<Ort::Value> inputs;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        for (const auto &tensor : tensors)
            if (tensor.type == Tensor::Float32)
                inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, (float *)tensor.data.get(),
                                                                 elements(tensor.shape), tensor.shape.data(),
                                                                 tensor.shape.size()));
            else if (tensor.type == Tensor::Int64)
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, (int64_t *)tensor.data.get(),
                                                                   elements(tensor.shape), tensor.shape.data(),
                                                                   tensor.shape.size()));

        return session.Run({}, inputNames.data(), inputs.data(), inputs.size(), outputNames.data(), outputNames.size());
    }
};

template <typename T>
struct Vec1
{
    const T *data;
    Size x;
    Vec1(const T *data, Size x) : data(data), x(x) {}
    const T &operator[](Size i) const { return data[i]; }
};

template <typename T>
struct Vec2
{
    const T *data;
    Size y, x;
    Vec2(const T *data, Size y, Size x) : data(data), y(y), x(x) {}
    Vec1<T> operator[](Size i) const { return Vec1<T>(data + i * x, x); }
};

template <typename T>
struct Vec3
{
    const T *data;
    Size z, y, x;
    Vec3(const T *data, Size z, Size y, Size x) : data(data), z(z), y(y), x(x) {}
    Vec2<T> operator[](Size i) const { return Vec2<T>(data + i * y * x, y, x); }
};

Detector::Detector(Type type, const String &path, Size device) : type(type)
{
    impl = Pointer<Impl>(new Impl(path, device));
}

Size Detector::imageSize() const { return impl->imageSize(); }

void detectYOLOv7(const Vector<Ort::Value> &values, Vector<Detections> &detections,  //
                  double width, double height, double threshold)
{
    detections.resize(values.size());
    for (Size i = 0; i < values.size(); ++i)
    {
        auto shape = values[i].GetTensorTypeAndShapeInfo().GetShape();
        auto output = values[i].GetTensorData<float>();
        Vec2<float> value(output, shape[0], shape[1]);
        Detections &list = detections[i];

        for (Size j = 0; j < value.y; ++j)
        {
            Detection result;
            auto ptr = value[j];
            result.x = ptr[1] / width;
            result.y = ptr[2] / height;
            result.w = (ptr[3] - ptr[1]) / width;
            result.h = (ptr[4] - ptr[2]) / height;
            std::tie(result.index, result.name) = getClass(ptr[5]);
            result.confidence = ptr[6];

            if (result.confidence >= threshold) list.push_back(result);
        }
    }
}

void detectYOLOv8(const Vector<Ort::Value> &values, Vector<Detections> &detections,  //
                  double width, double height, double threshold)
{
    auto shape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    auto output = values[0].GetTensorData<float>();
    Vec3<float> value(output, shape[0], shape[1], shape[2]);
    std::vector<float> transposed(value.x * value.y);
    Vec2<float> current(transposed.data(), value.x, value.y);

    detections.resize(value.z);
    for (Size i = 0; i < value.z; ++i)
    {
        std::unordered_map<Size, Detection> detmap;
        transposeMatrix(value[i].data, transposed.data(), value.y, value.x);

        for (Size j = 0; j < current.y; ++j)
        {
            auto ptr = current[j];
            auto it1 = std::max_element(ptr.data + 4, ptr.data + ptr.x);
            auto index = std::distance(ptr.data + 4, it1);
            auto confidence = *it1;
            if (confidence < threshold) continue;

            Detection result;
            result.x = (ptr[0] - ptr[2] / 2) / width;
            result.y = (ptr[1] - ptr[3] / 2) / height;
            result.w = ptr[2] / width;
            result.h = ptr[3] / height;
            std::tie(result.index, result.name) = getClass(index);
            result.confidence = confidence;

            auto it2 = detmap.find(result.index);
            if (it2 == detmap.end())
                detmap[result.index] = result;
            else if (it2->second.confidence < result.confidence)
                it2->second = result;
        }

        Detections &list = detections[i];
        list.reserve(detmap.size());
        for (const auto &pair : detmap) list.push_back(pair.second);
    }
}

void detectYOLOv10(const Vector<Ort::Value> &values, Vector<Detections> &detections,  //
                   double width, double height, double threshold)
{
    auto shape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    auto output = values[0].GetTensorData<float>();
    Vec3<float> value(output, shape[0], shape[1], shape[2]);

    detections.resize(value.z);
    for (int64_t i = 0; i < value.z; ++i)
    {
        Detections &list = detections[i];

        for (int64_t j = 0; j < value.y; ++j)
        {
            auto ptr = value[i][j];
            if (ptr[4] < threshold) continue;
            Detection result;
            result.x = ptr[0] / width;
            result.y = ptr[1] / height;
            result.w = (ptr[2] - ptr[0]) / width;
            result.h = (ptr[3] - ptr[1]) / height;
            std::tie(result.index, result.name) = getClass(ptr[5]);
            result.confidence = ptr[4];
            list.push_back(result);
        }
    }
}

void detectRTDETR(const Vector<Ort::Value> &values, Vector<Detections> &detections,  //
                  int64_t *dimensions, double width, double height, double threshold)
{
    auto shape = values[1].GetTensorTypeAndShapeInfo().GetShape();
    Vec2<int64_t> labels(values[0].GetTensorData<int64_t>(), shape[0], shape[1]);
    Vec2<float> scores(values[2].GetTensorData<float>(), shape[0], shape[1]);
    Vec3<float> boxes(values[1].GetTensorData<float>(), shape[0], shape[1], shape[2]);

    detections.resize(labels.y);
    for (int64_t i = 0; i < labels.y; ++i)
    {
        Detections &list = detections[i];
        auto current = dimensions + i * 2;
        double width = current[0], height = current[1];

        for (int64_t j = 0; j < labels.x; ++j)
        {
            if (scores[i][j] < threshold) continue;
            Detection result;
            auto ptr = boxes[i][j];
            result.x = ptr[0] / width;
            result.y = ptr[1] / height;
            result.w = (ptr[2] - ptr[0]) / width;
            result.h = (ptr[3] - ptr[1]) / height;
            std::tie(result.index, result.name) = getClass(labels[i][j]);
            result.confidence = scores[i][j];
            list.push_back(result);
        }
    }
}

Vector<Detections> Detector::operator()(Tensor images, Tensor dimensions, double threshold) const
{
    OBJDETEX_ASSERT(images, "Images tensor can't be empty");
    if (type == RT_DETR) OBJDETEX_ASSERT(dimensions, "Dimensions tensor can't be empty when running the RT-DETR model");
    if (images.shape.size() == 3) images.shape.insert(images.shape.begin(), 1);
    if (dimensions.shape.size() == 1) dimensions.shape.insert(dimensions.shape.begin(), 1);
    OBJDETEX_ASSERT(images.shape.size() == 4, "Images tensor must be 3D or 4D");
    if (type == RT_DETR) OBJDETEX_ASSERT(dimensions.shape.size() == 2, "Dimensions tensor must be 1D or 2D");

    Vector<Tensor> tensors;
    Vector<Sequence> inputs, outputs;

    if (type == YOLOv7)
        tensors = {images},       //
            inputs = {"images"},  //
            outputs = {"output"};
    else if (type == YOLOv8 || type == YOLOv9 || type == YOLOv10)
        tensors = {images},       //
            inputs = {"images"},  //
            outputs = {"output0"};
    else if (type == RT_DETR)
        tensors = {images, dimensions},                //
            inputs = {"images", "orig_target_sizes"},  //
            outputs = {"labels", "boxes", "scores"};
    else
        OBJDETEX_ASSERT(false, "Invalid detector type given");

    Vector<Detections> detections;
    auto values = (*impl)(tensors, inputs, outputs);
    double width = images.shape.at(2), height = images.shape.at(3);

    if (type == YOLOv7)
        detectYOLOv7(values, detections, width, height, threshold);
    else if (type == YOLOv8 || type == YOLOv9)
        detectYOLOv8(values, detections, width, height, threshold);
    else if (type == YOLOv10)
        detectYOLOv10(values, detections, width, height, threshold);
    else if (type == RT_DETR)
        detectRTDETR(values, detections, (int64_t *)dimensions.data.get(), width, height, threshold);
    else
        OBJDETEX_ASSERT(false, "Invalid detector type given");

    return detections;
}
}  // namespace ObjDetEx
