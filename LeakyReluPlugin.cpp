#include "LeakyReluPlugin.h"


LeakyReluPlugin::LeakyReluPlugin(const float negative_slope) {
	m_param.negative_slope = negative_slope;
}

LeakyReluPlugin::LeakyReluPlugin(const void* buffer, size_t size) {
	assert(size == sizeof(SERIALIZE_PARAM));
	const SERIALIZE_PARAM* pParam = reinterpret_cast<const SERIALIZE_PARAM*>(buffer);
	m_param.size = pParam->size;
	m_param.negative_slope = pParam->negative_slope;
}

LeakyReluPlugin::~LeakyReluPlugin() {

}

int LeakyReluPlugin::getNbOutputs() const {
	return 1;
}

nvinfer1::Dims LeakyReluPlugin::getOutputDimensions(int index,
		const nvinfer1::Dims* inputs, int nbInputDims) {
	assert(nbInputDims == 1);
	assert(index == 0);
	assert(inputs[index].nbDims == 3);
	return nvinfer1::DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int LeakyReluPlugin::initialize() {
	return 0;
}

void LeakyReluPlugin::terminate() {

}

size_t LeakyReluPlugin::getWorkspaceSize(int) const {
	return 0;
}

size_t LeakyReluPlugin::getSerializationSize() {
	return sizeof(SERIALIZE_PARAM);
}

void LeakyReluPlugin::serialize(void* buffer) {
	SERIALIZE_PARAM* pParam = reinterpret_cast<SERIALIZE_PARAM*>(buffer);
	pParam->size = m_param.size;
	pParam->negative_slope = m_param.negative_slope;
}

bool LeakyReluPlugin::supportsFormat(nvinfer1::DataType type,
		nvinfer1::PluginFormat format) const {
	return (type == nvinfer1::DataType::kFLOAT
			|| type == nvinfer1::DataType::kHALF)
			&& format == nvinfer1::PluginFormat::kNCHW;
}

void LeakyReluPlugin::configureWithFormat(const nvinfer1::Dims* inputDims,
		int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
		nvinfer1::DataType type, nvinfer1::PluginFormat format,
		int maxBatchSize) {
	m_param.size = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
}

