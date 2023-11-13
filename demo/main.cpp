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

#include <yolov7_cxx/yolov7_cxx.h>

#include <cassert>
#include <ranges>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

using Array = std::vector<float>;
using Shape = YOLOv7_CXX::Shape;

std::pair<Array, Shape> convert_image(const cv::Mat &image)
{
    Shape shape = {1, image.channels(), image.rows, image.cols};
    cv::Mat nchw = cv::dnn::blobFromImage(image, 1.0, {}, {}, true) / 255.f;
    Array array(nchw.ptr<float>(), nchw.ptr<float>() + nchw.total());
    return {array, shape};
}

void display_image(cv::Mat image, const std::vector<YOLOv7_CXX::Result> &detections)
{
    auto w = image.cols, h = image.rows;

    for (const auto &d : detections)
    {
        auto color = CV_RGB(255, 255, 255);  // CV_RGB(rand() % 256, rand() % 256, rand() % 256)
        auto name = d.name + ":" + std::to_string(int(d.confidence * 100)) + "%";
        cv::rectangle(image, cv::Rect(d.x * w, d.y * h, d.w * w, d.h * h), color);
        cv::putText(image, name, cv::Point(d.x * w, d.y * h), cv::FONT_HERSHEY_DUPLEX, 1, color);
    }

    cv::imshow("YOLOv7 Output", image);
}

void start(int argc, char* argv[])
{
    int64_t cuda_device = -1;
    std::string model_path;
    std::string video_source;
    std::string image_path;

    std::vector<std::string> args(argv, argv + argc);

    auto it = std::find(args.begin(), args.end(), "--cuda");
    if (it == args.end()) it = std::find(args.begin(), args.end(), "-c");
    if (it != args.end() && std::next(it) != args.end()) cuda_device = std::stoi(*std::next(it));

    it = std::find(args.begin(), args.end(), "--model");
    if (it == args.end()) it = std::find(args.begin(), args.end(), "-m");
    if (it != args.end() && std::next(it) != args.end()) model_path = *std::next(it);

    it = std::find(args.begin(), args.end(), "--video");
    if (it == args.end()) it = std::find(args.begin(), args.end(), "-v");
    if (it != args.end() && std::next(it) != args.end()) video_source = *std::next(it);

    it = std::find(args.begin(), args.end(), "--image");
    if (it == args.end()) it = std::find(args.begin(), args.end(), "-i");
    if (it != args.end() && std::next(it) != args.end()) image_path = *std::next(it);

    if (model_path.empty()) throw std::runtime_error("Model path can't be empty!");
    YOLOv7_CXX::YOLOv7 yolo(model_path, cuda_device);
    int image_size = yolo.image_size();

    if (!video_source.empty())
    {
        cv::VideoCapture cap;
        assert(cap.open(video_source));

        while (true)
        {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;
            assert(frame.channels() == 3);

            cv::resize(frame, frame, { image_size, image_size });
            auto [array, shape] = convert_image(frame);
            auto detections = yolo.detect(array.data(), shape);
            display_image(frame, detections[0]);
            if (cv::waitKey(1) >= 0) break;
        }
    }
    else if (!image_path.empty())
    {
        auto image = cv::imread(image_path);
        assert(!image.empty() && image.channels() == 3);
        cv::resize(image, image, { image_size, image_size });

        auto [array, shape] = convert_image(image);
        auto detections = yolo.detect(array.data(), shape);
        display_image(image, detections[0]);
        cv::waitKey(0);
    }
    else
        throw std::runtime_error("No input source is provided!");
}

int main(int argc, char *argv[])
{
    // Visual Studio debugger won't show exception message on unhandled exception
    try {
        start(argc, argv);
    } catch (std::exception& ex) {
        std::cout << "Exception: " << "\n"
            << ex.what() << '\n';
        throw;
    }
    return 0;
}
