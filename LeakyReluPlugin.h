#ifndef __LEAKY_RELU_PLUGIN_H__
#define __LEAKY_RELU_PLUGIN_H__

//#include "../../Utility/Define.h"
#include <memory>
#include <string.h>
#include <cstdint>
#include <cassert>

#include "NvInfer.h"
#include "NvCaffeParser.h"

class LeakyReluPlugin: public nvinfer1::IPluginExt {
public:
	//call when parse
	LeakyReluPlugin(const float negative_slope);
	//call when deserialize
	LeakyReluPlugin(const void* buffer, size_t size);
	~LeakyReluPlugin();

	//tell engine the output number
	int getNbOutputs() const override;

	//tell engine the dims of every output
	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
			int nbInputDims) override;

	//call when buildCudaEngine and deserializeCudaEngine
	int initialize() override;

	void terminate() override;

	//return the memory size of this layer, most return 0
	size_t getWorkspaceSize(int) const override;

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	// forword interface
	int enqueue(int batchSize, const void* const *inputs, void** outputs,
			void* workspace, cudaStream_t stream) override;

	//return the serialization size
	size_t getSerializationSize() override;

	// serialize the engine, then close everything down
	// trtModelStream = engine->serialize();
	void serialize(void* buffer) override;

	//Corresponding to configureWithFormat, supportsFormat will tell the \
	//engine which Format(fp32 or fp16) the plugin supports.
	virtual bool supportsFormat(nvinfer1::DataType type,
			nvinfer1::PluginFormat format) const;

	//When the user selects a certain Format,
	//the engine will call this interface to make specific settings for the fields in the plugin.
	virtual void configureWithFormat(const nvinfer1::Dims* inputDims,
			int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
			nvinfer1::DataType type, nvinfer1::PluginFormat format,
			int maxBatchSize);

protected:
	typedef struct {
		size_t size;
		float negative_slope;
	}SERIALIZE_PARAM;

	SERIALIZE_PARAM m_param;

//	size_t m_size;
//	float m_negative_slope;
};

#endif //__LEAKY_RELU_PLUGIN_H__

