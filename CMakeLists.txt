# BSD 3-Clause License
#
# Copyright (c) 2023, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.5)
project(YOLOV7_CXX LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(YOLOV7_CXX_MASTER OFF CACHE INTERNAL "")
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(YOLOV7_CXX_MASTER ON CACHE INTERNAL "")
endif()

set(ONNX_RUNTIME_PATH "/opt/onnxruntime" CACHE STRING "ONNX Runtime directory path")
set(CUDA_LIBRARIES_PATH "/opt/cuda11" CACHE STRING "CUDA libraries path compatible with the ONNX Runtime")

# Create Onnxruntime imported target
add_library(onnxruntime SHARED IMPORTED)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	set(ONNX_IMPORTED_LOCATION "${ONNX_RUNTIME_PATH}/lib/onnxruntime.dll")
	set(ONNX_IMPORTED_IMPLIB "${ONNX_RUNTIME_PATH}/lib/onnxruntime.lib")
	set(ONNX_INTERFACE_INCLUDE_DIRECTORIES "${ONNX_RUNTIME_PATH}/include")
	if(NOT EXISTS ${ONNX_IMPORTED_LOCATION})
		message(FATAL_ERROR "Excepted Onnxruntime library '${ONNX_IMPORTED_LOCATION}' does not exist. Please specify a valid ONNX_RUNTIME_PATH.(current: ${ONNX_RUNTIME_PATH})")
	endif()
	if(NOT EXISTS ${ONNX_IMPORTED_IMPLIB})
		message(FATAL_ERROR "Excepted Onnxruntime imported library '${ONNX_IMPORTED_IMPLIB}' does not exist. Please specify a valid ONNX_RUNTIME_PATH.(current: ${ONNX_RUNTIME_PATH})")
	endif()
	if(NOT EXISTS ${ONNX_INTERFACE_INCLUDE_DIRECTORIES})
		message(FATAL_ERROR "Excepted Onnxruntime include directory '${ONNX_INTERFACE_INCLUDE_DIRECTORIES}' does not exist. Please specify a valid ONNX_RUNTIME_PATH.(current: ${ONNX_RUNTIME_PATH})")
	endif()
	
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ONNX_IMPORTED_LOCATION}
        IMPORTED_IMPLIB   ${ONNX_IMPORTED_IMPLIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ONNX_INTERFACE_INCLUDE_DIRECTORIES}
    )
elseif(UNIX)
	set(ONNX_IMPORTED_LOCATION "${ONNX_RUNTIME_PATH}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}onnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}")
	set(ONNX_INTERFACE_INCLUDE_DIRECTORIES "${ONNX_RUNTIME_PATH}/include")
	if(NOT EXISTS ${ONNX_IMPORTED_LOCATION})
		message(FATAL_ERROR "Excepted Onnxruntime library '${ONNX_IMPORTED_LOCATION}' does not exist. Please specify a valid ONNX_RUNTIME_PATH.(current: ${ONNX_RUNTIME_PATH})")
	endif()
	if(NOT EXISTS ${ONNX_INTERFACE_INCLUDE_DIRECTORIES})
		message(FATAL_ERROR "Excepted Onnxruntime include directory '${ONNX_INTERFACE_INCLUDE_DIRECTORIES}' does not exist. Please specify a valid ONNX_RUNTIME_PATH.(current: ${ONNX_RUNTIME_PATH})")
	endif()
	
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ONNX_IMPORTED_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${ONNX_INTERFACE_INCLUDE_DIRECTORIES}
    )
else()
	message(FATAL_ERROR "Unsupported target platform.")
endif()


# Add yolov7_cxx library
file(GLOB SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/yolov7_cxx/*")
add_library(yolov7_cxx ${SOURCES})
target_include_directories(yolov7_cxx PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/")
target_link_directories(yolov7_cxx PUBLIC
    "${CUDA_LIBRARIES_PATH}/lib64")
target_link_libraries(yolov7_cxx PRIVATE
    onnxruntime)

# Add example executable
if(YOLOV7_CXX_MASTER)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
    add_executable(yolov7_demo demo/main.cpp)
    target_include_directories(yolov7_demo PRIVATE
        "${OpenCV_INCLUDE_DIRS}")
    target_link_libraries(yolov7_demo PRIVATE
        ${OpenCV_LIBS} yolov7_cxx)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_custom_command(TARGET yolov7_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:yolov7_demo> $<TARGET_FILE_DIR:yolov7_demo>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()
