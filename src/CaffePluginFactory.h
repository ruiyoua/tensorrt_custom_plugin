#ifndef __CAFFE_PLUGIN_FACTORY_H__
#define __CAFFE_PLUGIN_FACTORY_H__

#include <memory>
#include <map>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "LeakyReluPlugin.h"
#include "SlicePlugin.h"
#include "PReluPlugin.h"
#include "IPluginFactory.h"

class CaffePluginFactory: public IPluginFactory, public nvcaffeparser1::IPluginFactoryExt {
public:
	// caffe parser plugin implementation
	virtual bool isPlugin(const char* name) override;

	virtual bool isPluginExt(const char* name) override;

	virtual nvinfer1::IPlugin* createPlugin(const char* layerName,
			const nvinfer1::Weights* weights, int nbWeights) override;

	// deserialization plugin implementation
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName,
			const void* serialData, size_t serialLength) override;

	// User application destroys plugin when it is safe to do so.
	// Should be done after consumers of plugin (like ICudaEngine) are destroyed.
	virtual void destroyPlugin() override;

private:
	std::map< std::string, nvinfer1::IPlugin* > m_mapPlugins;
};

#endif //__PLUGIN_FACTORY_H__

