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

#include "yolov7_cxx.h"

#include <onnxruntime_cxx_api.h>

#include <locale>
#include <codecvt>
#include <string>
#include <memory>

namespace YOLOv7_CXX
{
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


#ifdef _WIN32   // ORTCHAR_T is wchar_t when _WIN32 is defined.
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

struct Impl
{
    Ort::Env env;
    Ort::SessionOptions options;
    Ort::Session session;

    Impl(const std::string &model_path, int64_t cuda_device)
        : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv7_CXX"),
          options(create_options(cuda_device)),
          session(env, TO_ONNX_STR(model_path), options)
    {
    }
};

int64_t elements(const std::vector<int64_t> &shape)
{
    int64_t target = 1;
    for (auto item : shape) target *= item;
    return target;
}

YOLOv7::YOLOv7(const std::string &model_path, int64_t cuda_device)
{
    impl = std::shared_ptr<Impl>(new Impl(model_path, cuda_device));
}

int64_t YOLOv7::image_size() const
{
    auto info = impl->session.GetInputTypeInfo(0);
    return info.GetTensorTypeAndShapeInfo().GetShape().back();
}

std::vector<Results> YOLOv7::detect(float *data, Shape shape)
{
    if (shape.size() == 3) shape.insert(shape.begin(), 1);
    double iw = shape.at(2), ih = shape.at(3);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input = Ort::Value::CreateTensor<float>(memory_info, data, elements(shape), shape.data(), shape.size());

    const char *input_names[] = {"images"};
    const char *output_names[] = {"output"};
    auto output = impl->session.Run({}, input_names, &input, 1, output_names, 1);

    std::vector<Results> results(output.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        shape = output[i].GetTensorTypeAndShapeInfo().GetShape();
        auto ptr = output[i].GetTensorData<float>();

        Results &temp = results[i];
        temp.resize(shape[0]);
        for (auto &d : temp)
        {
            d.x = ptr[1] / iw;
            d.y = ptr[2] / ih;
            d.w = (ptr[3] - ptr[1]) / iw;
            d.h = (ptr[4] - ptr[2]) / ih;
            d.index = ptr[5];
            d.name = class_names[d.index];
            d.confidence = ptr[6];
            ptr += shape[1];
        }
    }
    return results;
}
}  // namespace YOLOv7_CXX
