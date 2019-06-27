//
// base class for tensorrt inference
//

#ifndef __CV_INFERENCE_BASH_H__
#define __CV_INFERENCE_BASH_H__

//std inc
#include <fstream>
#include <vector>
#include <memory>
#include <iostream>
#include <numeric>

//tensorrt include
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "tensorrt_plugins/CaffePluginFactory.h"
#include "define.h"


class cvInferenceBase{
public:
	static bool GlobalInit();
	static bool GlobalFini();

public:
	virtual bool Init() = 0;
	virtual bool CreateTrtModel(const std::string& trt_model) = 0;	//pure virtual

	virtual bool OnTimer() { return true; }
	virtual bool Fini();
	virtual bool doInference();
protected:
	bool InitTrtEngine(const std::string& trt_model);
	bool FiniTrtEngine();

	virtual bool AllocBuffer();
	virtual bool FreeBuffer();

	bool saveTrtModel(nvinfer1::ICudaEngine* engine, const std::string& fileName);
	bool readTrtModel(const std::string& fileName, std::shared_ptr<char>& engine_buffer, int& engine_buffer_size);
public:
	//caffe model to tensorrt engine
	static nvinfer1::ICudaEngine* caffeToTRTModel(
					const std::string& deployFile,                 // name for caffe prototxt
					const std::string& modelFile,                  // name for model
					std::vector< std::string >& inputs,				// the inputs names
					const std::vector< std::string >& outputs,				// the outputs names
					unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
					IPluginFactory* pluginFactory, 				   // factory for plugin layers
					bool fp16 );


	//uff model to tensorrt engine
	static nvinfer1::ICudaEngine* uffToTRTModel(
					const std::string& uffFile,                    // name for uff model file
					const std::vector< std::string >& inputs,				// the inputs names
					const std::vector<nvinfer1::Dims>& inputDims,			// the inputs dims
					const std::vector< std::string >& outputs,				// the outputs names
			        unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
			        IPluginFactory* pluginFactory, 				   // factory for plugin layers
			        bool fp16);


	//onnx model to tensorrt engine
	static nvinfer1::ICudaEngine* onnxToTRTModel(
					const std::string& onnxModelFile,              // name for onnx model file
					std::vector< std::string >& inputs,				// the inputs names
					std::vector< std::string >& outputs,				// the outputs names
					unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
					bool fp16);

	static int volume(nvinfer1::Dims dims) {
	    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
	}

protected:
	cudaStream_t m_stream;

	nvinfer1::IRuntime* m_runtime;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;
	IPluginFactory* m_pluginFactory;

	static const char* TRT_DIRECTORY;

	std::vector<void*> m_gpu_buffers;		//gpu buffer

	//output/input name
	std::vector< std::string > m_input_name;
	std::vector< std::string > m_output_name;
	std::vector<int> m_input_index;
	std::vector<int> m_output_index;
	//network in/out
	std::vector< std::vector<float> > m_input;
	std::vector< nvinfer1::Dims > m_input_dims;
	std::vector< std::vector<float> > m_output;
	std::vector< nvinfer1::Dims > m_output_dims;
};

// Logger for TensorRT info/warning/errors
class Logger: public nvinfer1::ILogger {
public:

	Logger() :
			Logger(Severity::kWARNING) {
	}

	Logger(Severity severity) :
			reportableSeverity(severity) {
	}

	void log(Severity severity, const char* msg) override
	{
		// suppress messages with severity enum value greater than the reportable
		if (severity > reportableSeverity)
			return;

		switch (severity) {
		case Severity::kINTERNAL_ERROR:
			std::cerr << "INTERNAL_ERROR: ";
			break;
		case Severity::kERROR:
			std::cerr << "ERROR: ";
			break;
		case Severity::kWARNING:
			std::cerr << "WARNING: ";
			break;
		case Severity::kINFO:
			std::cerr << "INFO: ";
			break;
		default:
			std::cerr << "UNKNOWN: ";
			break;
		}
		std::cerr << msg << std::endl;
	}

	Severity reportableSeverity { Severity::kWARNING };
};

static Logger gLogger;


#endif
