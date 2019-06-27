#ifndef __IPLUGIN_FACTORY_H__
#define __IPLUGIN_FACTORY_H__

#include "NvInfer.h"

//the plugin factory interface class
class IPluginFactory : public nvinfer1::IPluginFactory {
public:
	//inherit from nvinfer1::IPluginFactory
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) = 0;

	//add destroy interface
	virtual void destroyPlugin() = 0;
};

#endif //__IPLUGIN_FACTORY_H__
