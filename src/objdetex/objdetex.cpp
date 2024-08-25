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

#define OBJDETEX_ASSERT(expr, msg)                                            \
    if (!static_cast<bool>(expr))                                             \
    {                                                                         \
        std::ostringstream stream;                                            \
        std::string file = __FILE__, func = __PRETTY_FUNCTION__;              \
        stream << "Assertion at " << file << ":" << __LINE__ << "->" << func; \
        stream << ":\n\t" << msg << std::endl;                                \
        throw Exception(stream.str());                                        \
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

Detector::Detector(Type type, const String &path, Size device) { impl = Pointer<Impl>(new Impl(path, device)); }

Size Detector::imageSize() const { return impl->imageSize(); }

void detectYOLOv7(const Vector<Ort::Value> &values, Vector<Detections> &detections,  //
                  double width, double height, double threshold)
{
    detections.resize(values.size());
    for (Size i = 0; i < values.size(); ++i)
    {
        auto shape = values[i].GetTensorTypeAndShapeInfo().GetShape();
        auto output = values[i].GetTensorData<float>();
        Detections &list = detections[i];

        for (Size j = 0; j < shape[0]; ++i)
        {
            Detection result;
            auto ptr = output + j * shape[1];
            result.x = ptr[1] / width;
            result.y = ptr[2] / height;
            result.w = (ptr[3] - ptr[1]) / width;
            result.h = (ptr[4] - ptr[2]) / height;
            std::tie(result.index, result.name) = getClass(ptr[5]);
            result.confidence = ptr[6];

            if (result.confidence >= threshold)  //
                list.push_back(result);
        }
    }
}

void detectYOLOv8(const Vector<Ort::Value> &values, Vector<Detections> &detections,  //
                  double width, double height, double threshold)
{
    auto outputShape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch = outputShape[0], features = outputShape[1], maximum = outputShape[2];
    auto output = values[0].GetTensorData<float>();

    detections.resize(batch);
    std::vector<float> transposed(maximum * features);
    for (Size i = 0; i < batch; ++i)
    {
        transposeMatrix(output + i * features * maximum, transposed.data(), features, maximum);
        std::unordered_map<Size, Detection> detmap;
        auto currentOutput = transposed.data();

        for (Size j = 0; j < maximum; ++j)
        {
            auto ptr = currentOutput + j * features;
            auto it1 = std::max_element(ptr + 4, ptr + features);
            auto index = std::distance(ptr + 4, it1);
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
    auto outputShape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch = outputShape[0], maximum = outputShape[1], features = outputShape[2];
    auto output = values[0].GetTensorData<float>();

    detections.resize(batch);
    for (int64_t i = 0; i < batch; ++i)
    {
        Detections &current = detections[i];
        auto curentOutput = output + i * maximum * features;

        for (int64_t j = 0; j < maximum; ++j)
        {
            auto ptr = curentOutput + j * features;
            if (ptr[4] < threshold) continue;
            Detection result;
            result.x = ptr[0] / width;
            result.y = ptr[1] / height;
            result.w = (ptr[2] - ptr[0]) / width;
            result.h = (ptr[3] - ptr[1]) / height;
            std::tie(result.index, result.name) = getClass(ptr[5]);
            result.confidence = ptr[4];
            current.push_back(result);
        }
    }
}

void detectRTDETR(const Vector<Ort::Value> &values, Vector<Detections> &detections,  //
                  int64_t *dimensions, double width, double height, double threshold)
{
    auto outputShape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch = outputShape[0], maximum = outputShape[1];
    auto labels = values[0].GetTensorData<int64_t>();
    auto boxes = values[1].GetTensorData<float>();
    auto scores = values[2].GetTensorData<float>();

    detections.resize(batch);
    for (int64_t i = 0; i < batch; ++i)
    {
        Detections &current = detections[i];
        auto curentScores = scores + i * maximum;
        auto currentLabels = labels + i * maximum;
        auto currentBoxes = boxes + i * maximum * 4;
        auto currentDims = dimensions + i * 2;

        double iw = currentDims[0], ih = currentDims[1];

        for (int64_t j = 0; j < maximum; ++j)
        {
            if (curentScores[j] < threshold) continue;
            Detection result;
            auto ptr = currentBoxes + j * 4;
            result.x = ptr[0] / iw;
            result.y = ptr[1] / ih;
            result.w = (ptr[2] - ptr[0]) / iw;
            result.h = (ptr[3] - ptr[1]) / ih;
            std::tie(result.index, result.name) = getClass(currentLabels[j]);
            result.confidence = curentScores[j];
            current.push_back(result);
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
    OBJDETEX_ASSERT(dimensions.shape.size() == 4, "Dimensions tensor must be 1D or 2D");

    Vector<Tensor> tensors;
    Vector<Sequence> inputs, outputs;

    if (type == YOLOv7)
        tensors = {images},       //
            inputs = {"images"},  //
            outputs = {"output"};
    else if (type == YOLOv8)
        tensors = {images},       //
            inputs = {"images"},  //
            outputs = {"output0"};
    else if (type == YOLOv9)
        tensors = {images},       //
            inputs = {"images"},  //
            outputs = {"output0"};
    else if (type == YOLOv10)
        tensors = {images},       //
            inputs = {"images"},  //
            outputs = {"output0"};
    else if (type == RT_DETR)
        tensors = {images, dimensions},                //
            inputs = {"images", "orig_target_sizes"},  //
            outputs = {"labels", "boxes", "scores"};

    auto values = (*impl)(tensors, inputs, outputs);

    Vector<Detections> detections;
    double width = images.shape.at(2), height = images.shape.at(3);

    if (type == YOLOv7)
        detectYOLOv7(values, detections, width, height, threshold);
    else if (type == YOLOv8 || type == YOLOv9)
        detectYOLOv8(values, detections, width, height, threshold);
    else if (type == YOLOv10)
        detectYOLOv10(values, detections, width, height, threshold);
    else if (type == RT_DETR)
        detectRTDETR(values, detections, (int64_t *)dimensions.data.get(), width, height, threshold);

    return detections;
}
}  // namespace ObjDetEx
