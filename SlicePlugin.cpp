#include <cuda.h>
#include <cuda_runtime_api.h>

#include "SlicePlugin.h"

SlicePlugin::SlicePlugin() {
	return;
}

SlicePlugin::SlicePlugin(const void* buffer, size_t size) {
	assert(size == (sizeof(m_nWidth) + sizeof(m_nHeight)));
	const int *d = static_cast<const int*>(buffer);
	m_nWidth = *d;
	m_nHeight = *(d + 1);

	return;
}

SlicePlugin::~SlicePlugin() {
	return;
}

int SlicePlugin::getNbOutputs() const {
	return PIECES_NUM;
}

nvinfer1::Dims SlicePlugin::getOutputDimensions(int index,
		const nvinfer1::Dims* inputs, int nbInputDims) {
	assert(index < PIECES_NUM);
	nvinfer1::Dims dims = nvinfer1::DimsCHW(inputs[0].d[0] / PIECES_NUM,
			inputs[0].d[1], inputs[0].d[2]);

	return dims;
}

int SlicePlugin::initialize() {
	return 0;
}

void SlicePlugin::terminate() {
	return;
}

size_t SlicePlugin::getWorkspaceSize(int) const {
	return 0;
}

int SlicePlugin::enqueue(int batchSize, const void* const *inputs,
		void** outputs, void* workspace, cudaStream_t stream) {
	//slice 4 channel to 2 pieces, every piece have 2 channel
	int element_count = m_nWidth * m_nHeight;
	int copy_size = 2 * element_count * sizeof(float);
	//Input is converted to a pointer to a const float
	const float* in = static_cast<const float*>(*inputs);
	//copy buffer, for every output,
	cudaMemcpy(static_cast<float*>(outputs[0]), in + 0 * element_count,
			copy_size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(static_cast<float*>(outputs[1]), in + 2 * element_count,
			copy_size, cudaMemcpyDeviceToDevice);

	return 0;
}

size_t SlicePlugin::getSerializationSize() {
	return sizeof(m_nWidth) + sizeof(m_nHeight);
}

//Slice serialization, save some member variables
void SlicePlugin::serialize(void* buffer) {
	//Serialize the buffer into 2 int type sizes
	int *d = static_cast<int*>(buffer);
	*d = m_nWidth;
	*(d + 1) = m_nHeight;
	return;
}

bool SlicePlugin::supportsFormat(nvinfer1::DataType type,
		nvinfer1::PluginFormat format) const {
	return (type == nvinfer1::DataType::kFLOAT)
			&& (format == nvinfer1::PluginFormat::kNCHW);
}

void SlicePlugin::configureWithFormat(const nvinfer1::Dims* inputDims,
		int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
		nvinfer1::DataType type, nvinfer1::PluginFormat format,
		int maxBatchSize) {
	m_nWidth = inputDims[0].d[1];
	m_nHeight = inputDims[0].d[2];

//	if (type == nvinfer1::DataType::kFLOAT) {
//		printf ("kFLOAT\n");
//	} else if (type == nvinfer1::DataType::kHALF) {
//		printf ("kHALF\n");
//	}

	return;
}

