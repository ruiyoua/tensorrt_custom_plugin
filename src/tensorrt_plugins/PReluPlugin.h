#ifndef __P_RELU_PLUGIN_H__
#define __P_RELU_PLUGIN_H__

#include <memory>
#include <string.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cassert>

#include "NvInfer.h"
#include "NvCaffeParser.h"

/*
	Prelu layer
	doesn't channel_shared_ (only one param),
	that is a case of Leaky ReLU ( you can implement it by nvinfer1::plugin::createPReLUPlugin)
*/


class PreluPlugin : public nvinfer1::IPluginExt
{
public:
	PreluPlugin(const nvinfer1::Weights *weights, int nbWeights);
	PreluPlugin(const void* buffer, size_t size);
	~PreluPlugin();
	int getNbOutputs() const override
    {
        return 1;
    };
	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims);
	int initialize() override;
	void terminate() override;
	size_t getWorkspaceSize(int) const override {  return 0; }

	size_t getSerializationSize() override;

	void serialize(void* buffer) override;

	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;

 	//与configureWithFormat对应，supportsFormat会告诉engine该plugin支持哪些Format(fp32 or fp16)
	virtual bool supportsFormat(nvinfer1::DataType type,
			nvinfer1::PluginFormat format) const;

	//当用户选定了某种Format，engine会调用此接口对plugin中的字段做具体设置
	virtual void configureWithFormat(const nvinfer1::Dims* inputDims,
			int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
			nvinfer1::DataType type, nvinfer1::PluginFormat format,
			int maxBatchSize);

    cudaError_t PReLUForward(const int count, const int channels, const int dim, const float* bottom_data,
      float* top_data, void* mDeviceKernel, const int div_factor);

protected:
	int input_c;
	int input_h;
	int input_w;
	int input_count;
	bool channel_shared_ {false};
	nvinfer1::Weights mWeights;
	void* mDeviceKernel{nullptr};

private:
	void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        cudaMalloc(&deviceData, count);
        cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice);
        return deviceData;
    }

    template<typename T> void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    size_t type2size(nvinfer1::DataType type) { return sizeof(float); }

    void convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights& weights)
    {
        memcpy(buffer, weights.values, weights.count * type2size(weights.type));
        buffer += weights.count * type2size(weights.type);
    }
};

#endif //__P_RELU_PLUGIN_H__

