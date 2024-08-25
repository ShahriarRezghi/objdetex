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

namespace ObjDetEx
{
using Sequence = const char *;

const char *class_names[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

#ifdef _WIN32  // ORTCHAR_T is wchar_t when _WIN32 is defined.
#define TO_ONNX_STR(stdStr) std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().from_bytes(stdStr).c_str()
#else
#define TO_ONNX_STR(stdStr) stdStr.c_str()
#endif

Ort::SessionOptions create_options(int64_t cuda_device)
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

void transpose_matrix(const float *input, float *output, int64_t x, int64_t y)
{
    for (int64_t i = 0; i < x; ++i)
        for (int64_t j = 0; j < y; ++j)  //
            output[j * x + i] = input[i * y + j];
}

struct Detector
{
    Ort::Env env;
    Ort::SessionOptions options;
    Ort::Session session;

    Detector(const String &modelPath, int64_t cudaDevice)
        : env(ORT_LOGGING_LEVEL_WARNING, "ObjDetEx"),
          options(create_options(cudaDevice)),
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

YOLOv7::YOLOv7(const String &modelPath, Size cudaDevice)
{
    impl = Pointer<Detector>(new Detector(modelPath, cudaDevice));
}

Size YOLOv7::imageSize() const { return impl->imageSize(); }

Vector<Detections> YOLOv7::operator()(Tensor image)
{
    if (image.shape.size() == 3) image.shape.insert(image.shape.begin(), 1);
    auto values = impl->operator()({image}, {"images"}, {"output"});
    double iw = image.shape.at(2), ih = image.shape.at(3);

    Vector<Detections> results(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        auto shape = values[i].GetTensorTypeAndShapeInfo().GetShape();
        auto ptr = values[i].GetTensorData<float>();

        Detections &temp = results[i];
        temp.resize(shape[0]);
        for (auto &d : temp)
        {
            d.x = ptr[1] / iw;
            d.y = ptr[2] / ih;
            d.w = (ptr[3] - ptr[1]) / iw;
            d.h = (ptr[4] - ptr[2]) / ih;
            // TODO add nearby int and clamping to index
            d.index = ptr[5];
            d.name = class_names[d.index];
            d.confidence = ptr[6];
            ptr += shape[1];
        }
    }
    return results;
}

YOLOv8::YOLOv8(const String &modelPath, Size cudaDevice)
{
    impl = Pointer<Detector>(new Detector(modelPath, cudaDevice));
}

Size YOLOv8::imageSize() const { return impl->imageSize(); }

Vector<Detections> YOLOv8::operator()(Tensor image, float threshold)
{
    if (image.shape.size() == 3) image.shape.insert(image.shape.begin(), 1);
    auto values = impl->operator()({image}, {"images"}, {"output0"});
    double iw = image.shape.at(2), ih = image.shape.at(3);

    auto outputShape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch = outputShape[0], features = outputShape[1], maximum = outputShape[2];
    auto output = values[0].GetTensorData<float>();

    std::vector<Detections> results(batch);
    std::vector<float> transposed(maximum * features);
    for (int64_t i = 0; i < batch; ++i)
    {
        transpose_matrix(output + i * features * maximum, transposed.data(), features, maximum);
        auto currentOutput = transposed.data();

        std::unordered_map<int64_t, Detection> detections;

        for (int64_t j = 0; j < maximum; ++j)
        {
            auto ptr = currentOutput + j * features;
            auto it1 = std::max_element(ptr + 4, ptr + features);
            auto index = std::distance(ptr + 4, it1);
            auto confidence = *it1;
            if (confidence < threshold) continue;

            Detection result;
            result.x = (ptr[0] - ptr[2] / 2) / iw;
            result.y = (ptr[1] - ptr[3] / 2) / ih;
            result.w = ptr[2] / iw;
            result.h = ptr[3] / ih;
            result.index = index;
            result.name = class_names[result.index];
            result.confidence = confidence;

            auto it2 = detections.find(result.index);
            if (it2 == detections.end())
                detections[result.index] = result;
            else if (it2->second.confidence < result.confidence)
                it2->second = result;
        }

        Detections &current = results[i];
        current.reserve(detections.size());
        for (const auto &pair : detections) current.push_back(pair.second);
    }
    return results;
}

YOLOv10::YOLOv10(const String &modelPath, Size cudaDevice)
{
    impl = Pointer<Detector>(new Detector(modelPath, cudaDevice));
}

Size YOLOv10::imageSize() const { return impl->imageSize(); }

Vector<Detections> YOLOv10::operator()(Tensor image, double threshold)
{
    if (image.shape.size() == 3) image.shape.insert(image.shape.begin(), 1);
    auto values = impl->operator()({image}, {"images"}, {"output0"});
    double iw = image.shape.at(2), ih = image.shape.at(3);

    auto outputShape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch = outputShape[0], maximum = outputShape[1], features = outputShape[2];
    auto output = values[0].GetTensorData<float>();

    std::vector<Detections> results(batch);
    for (int64_t i = 0; i < batch; ++i)
    {
        Detections &current = results[i];
        auto curentOutput = output + i * maximum * features;

        for (int64_t j = 0; j < maximum; ++j)
        {
            auto ptr = curentOutput + j * features;
            if (ptr[4] < threshold) continue;
            Detection result;
            result.x = ptr[0] / iw;
            result.y = ptr[1] / ih;
            result.w = (ptr[2] - ptr[0]) / iw;
            result.h = (ptr[3] - ptr[1]) / ih;
            result.index = ptr[5];
            result.name = class_names[result.index];
            result.confidence = ptr[4];
            current.push_back(result);
        }
    }
    return results;
}

RT_DETR::RT_DETR(const String &modelPath, Size cudaDevice)
{
    impl = Pointer<Detector>(new Detector(modelPath, cudaDevice));
}

Size RT_DETR::imageSize() const { return impl->imageSize(); }

Vector<Detections> RT_DETR::operator()(Tensor image, Tensor dims, double threshold)
{
    if (image.shape.size() == 3) image.shape.insert(image.shape.begin(), 1);
    if (dims.shape.size() == 1) dims.shape.insert(dims.shape.begin(), 1);
    auto values = impl->operator()({image, dims}, {"images", "orig_target_sizes"}, {"labels", "boxes", "scores"});

    auto outputShape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch = outputShape[0], maximum = outputShape[1];
    auto labels = values[0].GetTensorData<int64_t>();
    auto boxes = values[1].GetTensorData<float>();
    auto scores = values[2].GetTensorData<float>();

    std::vector<Detections> results(batch);
    for (int64_t i = 0; i < batch; ++i)
    {
        Detections &current = results[i];
        auto curentScores = scores + i * maximum;
        auto currentLabels = labels + i * maximum;
        auto currentBoxes = boxes + i * maximum * 4;
        auto currentDims = (int64_t *)dims.data.get() + i * 2;

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
            result.index = currentLabels[j];
            result.name = class_names[result.index];
            result.confidence = curentScores[j];
            current.push_back(result);
        }
    }
    return results;
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
}  // namespace ObjDetEx
