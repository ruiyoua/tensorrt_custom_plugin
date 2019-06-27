#include "PReluPlugin.h"

PreluPlugin::PreluPlugin(const nvinfer1::Weights *weights, int nbWeights) {
    assert(nbWeights==1);
    mWeights = weights[0];
    assert(mWeights.type == nvinfer1::DataType::kFLOAT || mWeights.type == nvinfer1::DataType::kHALF);
    mWeights.values = malloc(mWeights.count*type2size(mWeights.type));
    memcpy(const_cast<void*>(mWeights.values),weights[0].values,mWeights.count*type2size(mWeights.type));
}

PreluPlugin::PreluPlugin(const void* buffer, size_t size)
{
    const char* d = reinterpret_cast<const char*>(buffer), *a = d;
    read<int>(d,input_c);
    read<int>(d,input_h);
    read<int>(d,input_w);
    read<int>(d,input_count);
    read<bool>(d,channel_shared_);
    read<int64_t>(d,mWeights.count);
    read<nvinfer1::DataType>(d,mWeights.type);
    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));//deserializeToDevice(d,mDeviceKernel,mWeights.count);
    memcpy(const_cast<void*>(mWeights.values), d, mWeights.count * type2size(mWeights.type));
    d += mWeights.count * type2size(mWeights.type);
    assert(d == a + size);
}

PreluPlugin::~PreluPlugin()
{
    if (mWeights.values){
        free(const_cast<void*>(mWeights.values));
    }
}

nvinfer1::Dims PreluPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return nvinfer1::DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

bool PreluPlugin::supportsFormat(nvinfer1::DataType type,
		nvinfer1::PluginFormat format) const {
	return (type == nvinfer1::DataType::kFLOAT
			|| type == nvinfer1::DataType::kHALF)
			&& format == nvinfer1::PluginFormat::kNCHW;
}

void PreluPlugin::configureWithFormat(const nvinfer1::Dims* inputDims,
		int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
		nvinfer1::DataType type, nvinfer1::PluginFormat format,
		int maxBatchSize) {
    input_c = inputDims[0].d[0];
    input_h = inputDims[0].d[1];
    input_w = inputDims[0].d[2];
	input_count = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
	return;
}

size_t PreluPlugin::getSerializationSize() {
    return 4*sizeof(int) + sizeof(bool) + sizeof(mWeights.count)
    + sizeof(mWeights.type) +  mWeights.count * type2size(mWeights.type);
}

void PreluPlugin::serialize(void* buffer) {
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, input_c);
    write(d, input_h);
    write(d, input_w);
    write(d, input_count);
    write(d, channel_shared_);
    write(d, mWeights.count);
    write(d, mWeights.type);
    convertAndCopyToBuffer(d,mWeights);
    assert(d == a + getSerializationSize());
}

int PreluPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    const float *bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float *top_data = reinterpret_cast<float*>(outputs[0]);

    const int count = batchSize * input_count;
    const int dim = input_h*input_w;
    const int channels = input_c;
    const int div_factor = channel_shared_ ? channels : 1; //channel_shared_ default is false

    PReLUForward(count,channels,dim,bottom_data,top_data,mDeviceKernel,div_factor);

    return 0;
}

int PreluPlugin::initialize(){
    //std::cout << "~initialize  "<< mDeviceKernel << std::endl;
    cudaMalloc(&mDeviceKernel,mWeights.count*type2size(mWeights.type));
    cudaMemcpy(mDeviceKernel,mWeights.values,mWeights.count*type2size(mWeights.type),cudaMemcpyHostToDevice);
    return 0;
}

void PreluPlugin::terminate(){
    if (mDeviceKernel){
        //std::cout << "~terminate  "<< mDeviceKernel << std::endl;
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}
