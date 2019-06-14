#include "CaffePluginFactory.h"

// caffe parser plugin implementation
bool CaffePluginFactory::isPlugin(const char* name) {
	return isPluginExt(name);
}

inline bool IsPReluPlugin(const std::string& name) {
	if (name.find("prelu") == 0)
		return true;

	//arc face prelu
	if (name == "relu0")
		return true;
	if (name == "stage1_unit1_relu1")
		return true;
	if (name == "stage1_unit2_relu1")
		return true;
	if (name == "stage1_unit3_relu1")
		return true;
	if (name == "stage2_unit1_relu1")
		return true;
	if (name == "stage2_unit2_relu1")
		return true;
	if (name == "stage2_unit3_relu1")
		return true;
	if (name == "stage2_unit4_relu1")
		return true;
	if (name == "stage3_unit1_relu1")
		return true;
	if (name == "stage3_unit2_relu1")
		return true;
	if (name == "stage3_unit3_relu1")
		return true;
	if (name == "stage3_unit4_relu1")
		return true;
	if (name == "stage3_unit5_relu1")
		return true;
	if (name == "stage3_unit6_relu1")
		return true;
	if (name == "stage4_unit1_relu1")
		return true;
	if (name == "stage4_unit2_relu1")
		return true;
	if (name == "stage4_unit3_relu1")
		return true;

	return false;
}


inline bool IsLReluPlugin(const std::string& name) {
	if (name.find("lrelu") == 0)
		return true;

	return false;
}


inline bool IsSlicePlugin(const std::string& name) {
	if (name.find("slice") == 0)
		return true;

	return false;
}


bool CaffePluginFactory::isPluginExt(const char* name) {
	if (IsLReluPlugin(name))
		return true;
	if (IsPReluPlugin(name))
		return true;
	if (IsSlicePlugin(name))
		return true;

	return false;
}


nvinfer1::IPlugin* CaffePluginFactory::createPlugin(const char* layerName,
		const nvinfer1::Weights* weights, int nbWeights) {

	//ppn negative slope is 0.1
	if (IsLReluPlugin(layerName)) {
		m_mapPlugins[layerName] = (nvinfer1::IPlugin*)(new LeakyReluPlugin(0.1));
		return m_mapPlugins.at(layerName);
	}

	if (IsSlicePlugin(layerName)) {
		m_mapPlugins[layerName] = (nvinfer1::IPlugin*)(new SlicePlugin());
		return m_mapPlugins.at(layerName);
	}

	if (IsPReluPlugin(layerName)) {
		m_mapPlugins[layerName] = (nvinfer1::IPlugin*)(new PreluPlugin(weights, nbWeights));
		return m_mapPlugins.at(layerName);
	}

	return nullptr;
}


// deserialization plugin implementation
nvinfer1::IPlugin* CaffePluginFactory::createPlugin(const char* layerName,
		const void* serialData, size_t serialLength) {
	if (IsLReluPlugin(layerName)) {
		m_mapPlugins[layerName] = (nvinfer1::IPlugin*)(new LeakyReluPlugin(serialData, serialLength));
		return m_mapPlugins.at(layerName);
	}

	if (IsSlicePlugin(layerName)) {
		m_mapPlugins[layerName] = (nvinfer1::IPlugin*)(new SlicePlugin(serialData, serialLength));
		return m_mapPlugins.at(layerName);
	}

	if (IsPReluPlugin(layerName)) {
		m_mapPlugins[layerName] = (nvinfer1::IPlugin*)(new PreluPlugin(serialData, serialLength));
		return m_mapPlugins.at(layerName);
	}

	return nullptr;
}


// User application destroys plugin when it is safe to do so.
// Should be done after consumers of plugin (like ICudaEngine) are destroyed.
void CaffePluginFactory::destroyPlugin() {
    for (auto it = m_mapPlugins.begin(); it!=m_mapPlugins.end(); it++){
        if (IsLReluPlugin(it->first.c_str())){
            delete (LeakyReluPlugin*)(it->second);
        }
        else if (IsSlicePlugin(it->first.c_str())){
            delete (SlicePlugin*)(it->second);
        }
        else if (IsPReluPlugin(it->first.c_str())){
            delete (PreluPlugin*)(it->second);
        }
        m_mapPlugins.erase(it);
    }
}

