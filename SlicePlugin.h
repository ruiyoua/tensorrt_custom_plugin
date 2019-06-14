#ifndef __SLICE_PLUGIN_H__
#define __SLICE_PLUGIN_H__
#include <memory>
#include <string.h>
#include <cstdint>
#include <cassert>

#include "NvInfer.h"
#include "NvCaffeParser.h"

class SlicePlugin : public nvinfer1::IPluginExt
{
public:
	SlicePlugin();

	SlicePlugin(const void* buffer, size_t size);

    ~SlicePlugin();

    int getNbOutputs() const override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    //return 0
    size_t getWorkspaceSize(int) const override;

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    // forword
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;
    // serialize the engine, then close everything down
    // trtModelStream = engine->serialize();
    void serialize(void* buffer) override;

    virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const;

    virtual void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize);

protected:
    static const int PIECES_NUM {2};

	int m_nWidth;
    int m_nHeight;
};

#endif //__SLICE_PLUGIN_H__



