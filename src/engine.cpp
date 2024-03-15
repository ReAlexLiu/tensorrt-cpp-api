/*
* Copyright (c) 2018 L2Q All rights reserved.
*
* The Original Code and all software distributed under the License are
* distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
* EXPRESS OR IMPLIED, AND L2Q HEREBY DISCLAIMS ALL SUCH WARRANTIES,
* INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
* Please see the License for the specific language governing rights and
* limitations under the License.
*
* @Descripttion: Public Macro Definition
* @Author: l2q
* @Date: 2021/3/8 13:27
* @LastEditors: lucky
* @LastEditTime: 2023/4/7 8:15
 */
#include "engine.h"
#include "NvOnnxParser.h"
#include <algorithm>
#if __GNUC__ > 7
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/cudaimgproc.hpp>
#include <random>

using namespace nvinfer1;
using namespace Util;

std::vector<std::string> Util::getFilesInDirectory(const std::string& dirPath)
{
    std::vector<std::string> filepaths;
#if __GNUC__ > 7
    for (const auto& entry : std::filesystem::directory_iterator(dirPath))
    {
        filepaths.emplace_back(entry.path().string());
    }
#else
    for (const auto& entry : std::experimental::filesystem::directory_iterator(dirPath))
    {
        filepaths.emplace_back(entry.path());
    }
#endif
    return filepaths;
}

void Logger::log(Severity severity, const char* msg) noexcept
{
    switch (severity)
    {
    //! An internal error has occurred. Execution is unrecoverable.
    case Severity::kINTERNAL_ERROR:
        utility::xlogger::getInstance().log_(spdlog::level::critical, "", "", 0, msg);
        break;

        //! An application error has occurred.
    case Severity::kERROR:
        utility::xlogger::getInstance().log_(spdlog::level::err, "", "", 0, msg);
        break;

        //! An application error has been discovered, but TensorRT has recovered or fallen back to a default.
    case Severity::kWARNING:
        utility::xlogger::getInstance().log_(spdlog::level::warn, "", "", 0, msg);
        break;

        //!  Informational messages with instructional information.
    case Severity::kINFO:
        utility::xlogger::getInstance().log_(spdlog::level::info, "", "", 0, msg);
        break;

        //!  Verbose messages with debugging information.
    case Severity::kVERBOSE:
        utility::xlogger::getInstance().log_(spdlog::level::debug, "", "", 0, msg);
        break;

    default:
        utility::xlogger::getInstance().log_(spdlog::level::trace, "", "", 0, msg);
        break;
    }
}

Engine::Engine(const Options& options)
    : m_options(options)
{
}

bool Engine::build(std::string onnxModelPath, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals,
                   bool normalize)
{
    m_subVals    = subVals;
    m_divVals    = divVals;
    m_normalize  = normalize;

    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options, onnxModelPath);
    xinfo("Searching for engine file with name: {}", m_engineName);

    if (doesFileExist(m_engineName))
    {
        xinfo("Engine found, not regenerating...");
        return true;
    }

    if (!doesFileExist(onnxModelPath))
    {
        std::string errMsg = "Could not find model at path: " + onnxModelPath;
        xfatal(errMsg);
        throw std::runtime_error(errMsg);
    }

    // Was not able to find the engine file, generate...
    xinfo("Engine not found, generating. This could take a while...");

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder)
    {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch size is deprecated).
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network       = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser)
    {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
    std::ifstream   file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        xfatal("Unable to read engine file");
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed)
    {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1)
    {
        xfatal("Model needs at least 1 input!");
        throw std::runtime_error("Model needs at least 1 input!");
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i)
    {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch)
        {
            xfatal("The model has multiple inputs, each with differing batch sizes!");
            throw std::runtime_error("The model has multiple inputs, each with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    bool doesSupportDynamicBatch = false;
#endif
    if (input0Batch == -1)
    {
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
        doesSupportDynamicBatch = true;
#endif
        xinfo("Model supports dynamic batch size");
    }
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    else
#else
    else if (input0Batch == 1)
#endif
    {
        xinfo("Model only supports fixed batch size of {}", input0Batch);
        // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
        if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch)
        {
            std::string errMsg = "Model only supports a fixed batch size of " + std::to_string(input0Batch) +
                                 ". Must set Options.optBatchSize and Options.maxBatchSize to 1";
            xfatal(errMsg);
            throw std::runtime_error(errMsg);
        }
    }
#if !(NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    else
    {
        std::string errMsg = "Implementation currently only supports dynamic batch sizes or a fixed batch size of 1 (your batch size is fixed to " + std::to_string(input0Batch) + ")";
        xfatal(errMsg);
        throw std::runtime_error(errMsg);
    }
#endif

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    // Register a single optimization profile
    IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i)
    {
        // Must specify dimensions for all the inputs the model expects.
        const auto input     = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t    inputC    = inputDims.d[1];
        int32_t    inputH    = inputDims.d[2];
        int32_t    inputW    = inputDims.d[3];

        // Specify the optimization profile`
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
        if (doesSupportDynamicBatch)
        {
            optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        }
        else
        {
            optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        }
#else
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
#endif
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    if (m_options.precision == Precision::FP16)
    {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16())
        {
            xfatal("GPU does not support FP16 precision");
            throw std::runtime_error("GPU does not support FP16 precision");
        }
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (m_options.precision == Precision::INT8)
    {
        if (numInputs > 1)
        {
            xfatal("This implementation currently only supports INT8 quantization for single input models");
            throw std::runtime_error("This implementation currently only supports INT8 quantization for single input models");
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8())
        {
            xfatal("GPU does not support INT8 precision");
            throw std::runtime_error("GPU does not support INT8 precision");
        }

        // Ensure the user has provided path to calibration data directory
        if (m_options.calibrationDataDirectoryPath.empty())
        {
            xfatal("If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
            throw std::runtime_error("If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
        }

        config->setFlag((BuilderFlag::kINT8));

        const auto input               = network->getInput(0);
        const auto inputName           = input->getName();
        const auto inputDims           = input->getDimensions();
        const auto calibrationFileName = m_engineName + ".calibration";

        m_calibrator                   = std::make_unique<Int8EntropyCalibrator2>(m_options.calibrationBatchSize, inputDims.d[3],
                                                                                  inputDims.d[2], m_options.calibrationDataDirectoryPath,
                                                                                  calibrationFileName, inputName, subVals, divVals,
                                                                                  normalize);
        config->setInt8Calibrator(m_calibrator.get());
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    std::unique_ptr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
    if (!plan)
    {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    xinfo("Success, saved engine to {}", m_engineName);

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

Engine::~Engine()
{
    // Free the GPU memory
    for (auto& buffer : m_buffers)
    {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

bool Engine::loadNetwork()
{
    // Read the serialized model from disk
    std::ifstream   file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        xfatal("Unable to read engine file");
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<IRuntime>{ createInferRuntime(m_logger) };
    if (!m_runtime)
    {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                      ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        xfatal(errMsg);
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine)
    {
        return false;
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context)
    {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    m_buffers.resize(m_engine->getNbIOTensors());
#else
    // TODO: rollback libnvinfer API to TensorRT 8.2.1.9
    // TODO: Needs a way to dynamically get input and output buffer size
    int32_t nb_io_tensors = 2;
    m_buffers.resize(nb_io_tensors);
    for (int i = 0; i < nb_io_tensors; ++i)
    {
        m_IOTensorNames.emplace_back(m_engine->getBindingName(i));
    }
#endif
    // Create a cuda stream
    // Allocate GPU memory for input and output buffers
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));
#if !(NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    // Define
    // Allocate input
    size_t input_mem_size = m_options.maxBatchSize *
                            m_options.channels * m_options.width * m_options.height * sizeof(float);
    // cudaMemcpyAsync
    checkCudaErrorCode(cudaMallocManaged(&m_buffers[0], input_mem_size));
    checkCudaErrorCode(cudaStreamAttachMemAsync(stream, m_buffers[0], 0, cudaMemAttachGlobal));
    m_inputDims.emplace_back(m_options.channels, m_options.width, m_options.height);
#endif
    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();

#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i)
    {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType  = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        if (tensorType == TensorIOMode::kINPUT)
        {
            // Allocate memory for the input
            // Allocate enough to fit the max batch size (we could end up using less later)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float), stream));

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            m_inputBatchSize = tensorShape.d[0];
        }
        else if (tensorType == TensorIOMode::kOUTPUT)
        {
            // The binding is an output
            uint32_t outputLenFloat = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j)
            {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream));
        }
        else
        {
            xfatal("IO Tensor is neither an input or output!");
            throw std::runtime_error("IO Tensor is neither an input or output!");
        }
    }
#else
    nvinfer1::Dims3 out_tensor_shape = { m_options.maxBatchSize, m_options.totals, m_options.targets };
    m_outputDims.push_back(out_tensor_shape);
    uint32_t outputLenFloat = 1;
    for (int j = 1; j < out_tensor_shape.nbDims; ++j)
    {
        // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
        outputLenFloat *= out_tensor_shape.d[j];
    }
    m_outputLengthsFloat.push_back(outputLenFloat);
    // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
    size_t output_mem_size = m_options.maxBatchSize * outputLenFloat * sizeof(float);
    checkCudaErrorCode(cudaMallocManaged(&m_buffers[1], output_mem_size));
    checkCudaErrorCode(cudaStreamAttachMemAsync(stream, m_buffers[1], 0, cudaMemAttachGlobal));
#endif

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

bool Engine::runInference(const std::vector<std::vector<cv::cuda::GpuMat>>& inputs, std::vector<std::vector<std::vector<float>>>& featureVectors)
{
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty())
    {
        xerror("Provided input vector is empty!");
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs)
    {
        xerror("Incorrect number of inputs provided!");
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize))
    {
        xerror("The batch size is larger than the model expects! Model max batch size: {}, Batch size provided to call to runInference: {}", m_options.maxBatchSize, inputs[0].size());
        return false;
    }

#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    // Ensure that if the model has a fixed batch size that is greater than 1, the input has the correct length
    if (m_inputBatchSize != -1 && inputs[0].size() != static_cast<size_t>(m_inputBatchSize))
    {
        xerror("The batch size is different from what the model expects! Model batch size: {}, Batch size provided to call to runInference: {}", m_inputBatchSize, inputs[0].size());
        return false;
    }
#endif

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i)
    {
        if (inputs[i].size() != static_cast<size_t>(batchSize))
        {
            xerror("The batch size needs to be constant for all inputs!");
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i)
    {
        const auto& batchInput = inputs[i];
        const auto& dims       = m_inputDims[i];

        auto&       input      = batchInput[0];
        if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2])
        {
            xerror("Input does not have correct size! Expected: ({}, {}, {}), Got: ({}, {}, {}), Ensure you resize your input image to the correct size",
                   dims.d[0], dims.d[1], dims.d[2], input.channels(), input.rows, input.cols);
            return false;
        }

        nvinfer1::Dims4 inputDims = { batchSize, dims.d[0], dims.d[1], dims.d[2] };
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
        m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);  // Define the batch size
#else
        int32_t input_idx = m_engine->getBindingIndex(m_IOTensorNames[i].c_str());
        m_context->setBindingDimensions(input_idx, inputDims);
#endif

        // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format.
        // The following method converts NHWC to NCHW.
        // Even though TensorRT expects NCHW at IO, during optimization, it can internally use NHWC to optimize cuda kernels
        // See: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        auto  mfloat      = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
        auto* dataPointer = mfloat.ptr<void>();

        checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], dataPointer,
                                           mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float),
                                           cudaMemcpyDeviceToDevice, inferenceCudaStream));
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified())
    {
        xfatal("Not all required dimensions specified.");
        throw std::runtime_error("Not all required dimensions specified.");
    }

#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i)
    {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status)
        {
            return false;
        }
    }
#endif
    // Run inference.
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    bool status = m_context->enqueueV3(inferenceCudaStream);
#else
    void* bindings[] = { m_buffers[0], m_buffers[1] };
    bool  status     = m_context->enqueueV2(bindings, inferenceCudaStream, nullptr);
#endif
    if (!status)
    {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    for (int batch = 0; batch < batchSize; ++batch)
    {
        // Batch
        std::vector<std::vector<float>> batchOutputs{};
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding)
        {
            // We start at index m_inputDims.size() to account for the inputs in our m_buffers
            std::vector<float> output;
            auto               outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
            output.resize(outputLenFloat);
            // Copy the output
            checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}

cv::cuda::GpuMat Engine::blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batchInput, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize)
{
    cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

    size_t           width = batchInput[0].cols * batchInput[0].rows;
    for (size_t img = 0; img < batchInput.size(); img++)
    {
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                             &(gpu_dst.ptr()[width * 2 + width * 3 * img]))
        };
        cv::cuda::split(batchInput[img], input_channels);  // HWC -> CHW
    }

    cv::cuda::GpuMat mfloat;
    if (normalize)
    {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    }
    else
    {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

std::string Engine::serializeEngineOptions(const Options& options, const std::string& onnxModelPath)
{
    //const auto               filenamePos = onnxModelPath.find_last_of('/') + 1;
    //std::string              engineName  = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

    std::string              engineName = onnxModelPath.substr(0, onnxModelPath.find_last_of('.')) + ".engine";

    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size())
    {
        xfatal("Provided device index is out of range!");
        throw std::runtime_error("Provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName += "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16)
    {
        engineName += ".fp16";
    }
    else if (options.precision == Precision::FP32)
    {
        engineName += ".fp32";
    }
    else
    {
        engineName += ".int8";
    }

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);

    return engineName;
}

void Engine::getDeviceNames(std::vector<std::string>& deviceNames)
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device = 0; device < numGPUs; device++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor)
{
    int   unpad_w, unpad_h;
    float r_w = width / (input.cols * 1.0);
    float r_h = height / (input.rows * 1.0);
    if (r_h > r_w)
    {
        unpad_w = width;
        unpad_h = r_w * input.rows;
    }
    else
    {
        unpad_w = r_h * input.cols;
        unpad_h = height;
    }
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadCenter(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor)
{
    int   unpad_w, unpad_h, unpad_x, unpad_y;
    float r_w = width / (input.cols * 1.0);
    float r_h = height / (input.rows * 1.0);
    if (r_h > r_w)
    {
        unpad_w = width;
        unpad_h = r_w * input.rows;
        unpad_x = 0;
        unpad_y = (height - unpad_h) / 2;
    }
    else
    {
        unpad_w = r_h * input.cols;
        unpad_h = height;
        unpad_x = (width - unpad_w) / 2;
        unpad_y = 0;
    }
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(unpad_x, unpad_y, re.cols, re.rows)));
    return out;
}

//左上顶点补边方式坐标还原
void Engine::resetLocationRightBottom(float resizeRatio, unsigned int width, unsigned int height, cv::Rect_<float> &bbox)
{
    bbox.x      = bbox.x * resizeRatio;
    bbox.y      = bbox.y * resizeRatio;
    bbox.width  = bbox.width * resizeRatio;
    bbox.height = bbox.height * resizeRatio;


    bbox.x      = (bbox.x < width) ? bbox.x : width;
    bbox.y      = (bbox.y < height) ? bbox.y : height;
    bbox.width  = (bbox.width < width) ? bbox.width : width;
    bbox.height = (bbox.height < height) ? bbox.height : height;

    bbox.x      = (bbox.x >= 0) ? bbox.x : 0;
    bbox.y      = (bbox.y >= 0) ? bbox.y : 0;
    bbox.width  = (bbox.width >= 0) ? bbox.width : 0;
    bbox.height = (bbox.height >= 0) ? bbox.height : 0;
}

//中心补边方式坐标还原
void Engine::resetLocationCenter(float resizeRatio, unsigned int width, unsigned int height, unsigned int input_w, unsigned int input_h, cv::Rect_<float> &bbox)
{
    int   w, h, x, y;
    float r_w = input_w / (width * 1.0);
    float r_h = input_h / (height * 1.0);
    if (r_h > r_w)
    {
        w = input_w;
        h = r_w * height;
        x = 0;
        y = (input_h - h) / 2;
        bbox.y -= y;
    }
    else
    {
        w = r_h * width;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
        bbox.x -= x;
    }

    bbox.x      = bbox.x * resizeRatio;
    bbox.y      = bbox.y * resizeRatio;
    bbox.width  = bbox.width * resizeRatio;
    bbox.height = bbox.height * resizeRatio;

    bbox.x      = (bbox.x < width) ? bbox.x : width;
    bbox.y      = (bbox.y < height) ? bbox.y : height;
    bbox.width  = (bbox.width < width) ? bbox.width : width;
    bbox.height = (bbox.height < height) ? bbox.height : height;

    bbox.x      = (bbox.x >= 0) ? bbox.x : 0;
    bbox.y      = (bbox.y >= 0) ? bbox.y : 0;
    bbox.width  = (bbox.width >= 0) ? bbox.width : 0;
    bbox.height = (bbox.height >= 0) ? bbox.height : 0;
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output)
{
    if (input.size() != 1)
    {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output)
{
    if (input.size() != 1 || input[0].size() != 1)
    {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH,
                                               const std::string&          calibDataDirPath,
                                               const std::string&          calibTableName,
                                               const std::string&          inputBlobName,
                                               const std::array<float, 3>& subVals,
                                               const std::array<float, 3>& divVals,
                                               bool                        normalize,
                                               bool                        readCache)
    : m_batchSize(batchSize)
    , m_inputW(inputW)
    , m_inputH(inputH)
    , m_imgIdx(0)
    , m_calibTableName(calibTableName)
    , m_inputBlobName(inputBlobName)
    , m_subVals(subVals)
    , m_divVals(divVals)
    , m_normalize(normalize)
    , m_readCache(readCache)
{
    // Allocate GPU memory to hold the entire batch
    m_inputCount = 3 * inputW * inputH * batchSize;
    checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

    // Read the name of all the files in the specified directory.
    if (!doesFileExist(calibDataDirPath))
    {
        auto errMsg = "Directory at provided path does not exist: " + calibDataDirPath;
        xfatal(errMsg);
        throw std::runtime_error(errMsg);
    }

    m_imgPaths = getFilesInDirectory(calibDataDirPath);
    if (m_imgPaths.size() < static_cast<size_t>(batchSize))
    {
        xfatal("There are fewer calibration images than the specified batch size!");
        throw std::runtime_error("There are fewer calibration images than the specified batch size!");
    }

    // Randomize the calibration data
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH > 8201)
    auto rd  = std::random_device{};
    auto rng = std::default_random_engine{ rd() };
#else
    auto rng = std::default_random_engine{};
#endif
    std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    // Return the batch size
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void** bindings, const char** names, int32_t nbBindings) noexcept
{
    // This method will read a batch of images into GPU memory, and place the pointer to the GPU memory in the bindings variable.

    if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size()))
    {
        // There are not enough images left to satisfy an entire batch
        return false;
    }

    // Read the calibration images into memory for the current batch
    std::vector<cv::cuda::GpuMat> inputImgs;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++)
    {
        xinfo("Reading image {}: {}", i, m_imgPaths[i]);
        auto cpuImg = cv::imread(m_imgPaths[i]);
        if (cpuImg.empty())
        {
            xerror("Unable to read image at path: {}", m_imgPaths[i]);
            return false;
        }

        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);
        cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

        // TODO: Define any preprocessing code here, such as resizing
        auto resized = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH, m_inputW);

        inputImgs.emplace_back(std::move(resized));
    }

    // Convert the batch from NHWC to NCHW
    // ALso apply normalization, scaling, and mean subtraction
    auto  mfloat      = Engine::blobFromGpuMats(inputImgs, m_subVals, m_divVals, m_normalize);
    auto* dataPointer = mfloat.ptr<void>();

    // Copy the GPU buffer to member variable so that it persists
    checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice));

    m_imgIdx += m_batchSize;
    if (std::string(names[0]) != m_inputBlobName)
    {
        xerror("Incorrect input name provided!");
        return false;
    }
    bindings[0] = m_deviceInput;
    return true;
}

void const* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
    xinfo("Searching for calibration cache: {}", m_calibTableName);
    m_calibCache.clear();
    std::ifstream input(m_calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (m_readCache && input.good())
    {
        xinfo("Reading calibration cache: {}", m_calibTableName);
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
    }
    length = m_calibCache.size();
    return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* ptr, std::size_t length) noexcept
{
    xinfo("Writing calib cache: {} Size: {} bytes", m_calibTableName, length);
    std::ofstream output(m_calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(ptr), length);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    checkCudaErrorCode(cudaFree(m_deviceInput));
}
