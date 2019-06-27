#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "cvInferenceBase.h"

const char* cvInferenceBase::TRT_DIRECTORY = "./trt_models";

bool cvInferenceBase::GlobalInit() {
	//more plugin support
	initLibNvInferPlugins(&gLogger, "");

	//create trt models derectory
	if (access(TRT_DIRECTORY, 0) != 0) {
		CHECK_EQ(mkdir(TRT_DIRECTORY, 0755), 0);
	}

	return true;
}

bool cvInferenceBase::GlobalFini() {
	//shut down protobuf lib
	nvcaffeparser1::shutdownProtobufLibrary();
}

bool cvInferenceBase::InitTrtEngine(const std::string& trt_model) {
	//read engine from file
	std::shared_ptr<char> engine_buffer;
	int engine_buffer_size;
	CHECK(readTrtModel(trt_model, engine_buffer, engine_buffer_size));

	//init engine
	m_runtime = nvinfer1::createInferRuntime(gLogger);
	m_engine = m_runtime->deserializeCudaEngine(engine_buffer.get(),
			engine_buffer_size, m_pluginFactory);
	m_context = m_engine->createExecutionContext();

	return true;
}

bool cvInferenceBase::Fini() {
	cvInferenceBase::FreeBuffer();
	cvInferenceBase::FiniTrtEngine();
	return true;
}

bool cvInferenceBase::FiniTrtEngine() {

	// Destroy the engine
	m_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();

	m_context = nullptr;
	m_engine = nullptr;
	m_runtime = nullptr;

	// Destroy plugins created by factory
	if (m_pluginFactory != nullptr) {
		m_pluginFactory->destroyPlugin();
		delete m_pluginFactory;
		m_pluginFactory = nullptr;
	}

	return true;
}

bool cvInferenceBase::AllocBuffer() {
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	CHECK_EQ(m_engine->getNbBindings(), m_input_name.size() + m_output_name.size());

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	m_gpu_buffers.resize(m_input_name.size() + m_output_name.size());
	m_input_index.resize(m_input_name.size());
	m_input_dims.resize(m_input_name.size());
	m_input.resize(m_input_name.size());

	m_output_index.resize(m_output_name.size());
	m_output_dims.resize(m_output_name.size());
	m_output.resize(m_output_name.size());

	// create GPU buffers and cpu buffers
	for (int i = 0; i < m_input_name.size(); i++) {
		m_input_index[i] = (m_engine->getBindingIndex(m_input_name[i].c_str()));
		m_input_dims[i] = m_engine->getBindingDimensions((int) m_input_index[i]);

		//alloc
		m_input[i].resize(cvInferenceBase::volume(m_input_dims[i]));
		CHECK( cudaSuccess == cudaMalloc(&m_gpu_buffers[m_input_index[i]],
				cvInferenceBase::volume(m_input_dims[i]) * sizeof(float)));
	}

	for (int i = 0; i < m_output_name.size(); i++) {
		m_output_index[i]= (m_engine->getBindingIndex(m_output_name[i].c_str()));
		m_output_dims[i] = m_engine->getBindingDimensions((int) m_output_index[i]);

		//alloc
		m_output[i].resize(cvInferenceBase::volume(m_output_dims[i]));
		CHECK( cudaSuccess == cudaMalloc(&m_gpu_buffers[m_output_index[i]],
				cvInferenceBase::volume(m_output_dims[i]) * sizeof(float)));
	}

	// create a stream
	CHECK( cudaSuccess == cudaStreamCreate(&m_stream));

	return true;
}


bool cvInferenceBase::FreeBuffer() {
	// release the stream and the buffers
	cudaStreamDestroy(m_stream);
	CHECK_EQ(m_input_name.size(), m_input_index.size());
	CHECK_EQ(m_output_name.size(), m_output_index.size());
	for (int i = 0; i < m_input_name.size(); i++) {
		m_input[i].resize(0);
		CHECK(cudaSuccess == cudaFree(m_gpu_buffers[m_input_index[i]]));
	}
	m_input.resize(0);
	for (int i = 0; i < m_output_name.size(); i++) {
		m_output[i].resize(0);
		CHECK(cudaSuccess == cudaFree(m_gpu_buffers[m_output_index[i]]));
	}
	m_output.resize(0);
	return true;
}

bool cvInferenceBase::doInference() {
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	for (int i = 0; i < m_input_name.size(); i++) {
		CHECK_EQ( cudaSuccess,
				cudaMemcpyAsync(m_gpu_buffers[m_input_index[i]], (void*)&m_input[i][0],
				cvInferenceBase::volume(m_input_dims[i]) * sizeof(float),
				cudaMemcpyHostToDevice, m_stream));
	}

	m_context->enqueue(1, &m_gpu_buffers[0], m_stream, nullptr);
	for (int i = 0; i < m_output_name.size(); i++) {
		CHECK_EQ( cudaSuccess,
				cudaMemcpyAsync((void*)&m_output[i][0], m_gpu_buffers[m_output_index[i]],
				cvInferenceBase::volume(m_output_dims[i]) * sizeof(float),
				cudaMemcpyDeviceToHost, m_stream));
	}

	cudaStreamSynchronize(m_stream);
}

nvinfer1::ICudaEngine* cvInferenceBase::caffeToTRTModel(
				const std::string& deployFile,                 // name for caffe prototxt
				const std::string& modelFile,                  // name for model
				std::vector< std::string >& inputs,				// the inputs names
				const std::vector< std::string >& outputs,				// the outputs names
				unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
				IPluginFactory* pluginFactory, 				   // factory for plugin layers
				bool fp16 ) {
    // create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

	//extra plugin support
	if (pluginFactory) {
		nvcaffeparser1::IPluginFactoryExt* caffePluginFactory = dynamic_cast<nvcaffeparser1::IPluginFactoryExt*>(pluginFactory);
		IF_NOT (caffePluginFactory) return nullptr;
		parser->setPluginFactoryExt(caffePluginFactory);
	}

    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
    		deployFile.c_str(), modelFile.c_str(), *network,
            fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);

    IF_NOT (nullptr != blobNameToTensor)
        return nullptr;

    // specify which tensors are outputs
    for (auto& s : outputs) {
        IF_NOT (blobNameToTensor->find(s.c_str()) != nullptr) {
            std::cout << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
    	nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
    	inputs.push_back(network->getInput(i)->getName());
    	std::cout << "Caffe Input \"" << network->getInput(i)->getName() << "\": "
        		<< dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
    	nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getOutput(i)->getDimensions());
//    	outputs.push_back(network->getOutput(i)->getName());
    	std::cout << "Caffe Output \"" << network->getOutput(i)->getName() << "\": "
        		<< dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    //configure Builder
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1ULL << 30);			//max gpu memory 1 Giga
	builder->setFp16Mode(fp16);

	std::cout << "building cuda engine......." << std::endl;

	// Build the engine
    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
    IF_NOT (engine != nullptr)
    	std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();

    return engine;
}

nvinfer1::ICudaEngine* cvInferenceBase::uffToTRTModel(
				const std::string& uffFile,                    // name for uff model file
				const std::vector< std::string >& inputs,				// the inputs names
				const std::vector<nvinfer1::Dims>& inputDims,			// the inputs dims
				const std::vector< std::string >& outputs,				// the outputs names
		        unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
		        IPluginFactory* pluginFactory, 				   // factory for plugin layers
		        bool fp16) {
    // create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
	//extra plugin support
	if (pluginFactory) {
		nvuffparser::IPluginFactoryExt* uffPluginFactory = dynamic_cast<nvuffparser::IPluginFactoryExt*>(pluginFactory);
		IF_NOT (uffPluginFactory) return nullptr;
		parser->setPluginFactoryExt(uffPluginFactory);
	}

    // specify which tensors are outputs
    for (auto& s : outputs) {
        IF_NOT (parser->registerOutput(s.c_str()))  {
        	std::cout << "Failed to register output " << s << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    CHECK_EQ (inputs.size(), inputDims.size());
    for (int i = 0; i < inputs.size(); i++) {
        if (!parser->registerInput(inputs[i].c_str(), inputDims[i], nvuffparser::UffInputOrder::kNCHW)) {
        	std::cout << "Failed to register input " << inputs[i] << std::endl;
            return nullptr;
        }
    }

    if (!parser->parse(uffFile.c_str(), *network, fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT))
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
    	nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
//    	inputs.push_back(network->getInput(i)->getName());
        std::cout << "Uff Input \"" << network->getInput(i)->getName() << "\": "
        		<< dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
    	nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getOutput(i)->getDimensions());
//    	outputs.push_back(network->getOutput(i)->getName());
    	std::cout << "Uff Output \"" << network->getOutput(i)->getName() << "\": "
        		<< dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }


    //configure Builder
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1ULL << 30);			//max gpu memory 1 Giga
	builder->setFp16Mode(fp16);

    // Build the engine
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
    IF_NOT (engine != nullptr)
		std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}


// ONNX is not supported in Windows
nvinfer1::ICudaEngine* cvInferenceBase::onnxToTRTModel(
				const std::string& onnxModelFile,              // name for onnx model file
				std::vector< std::string >& inputs,				// the inputs names
				std::vector< std::string >& outputs,				// the outputs names
				unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
				bool fp16) {
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // parse the onnx model to populate the network, then set the outputs
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    IF_NOT (parser->parseFromFile(onnxModelFile.c_str(), verbosity)) {
    	std::cout << "failed to parse onnx file" << std::endl;
        return nullptr;
    }

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
    	nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
    	inputs.push_back(network->getInput(i)->getName());
    	std::cout << "Onnx Input \"" << network->getInput(i)->getName() << "\": "
        		<< dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
    	nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getOutput(i)->getDimensions());
    	outputs.push_back(network->getOutput(i)->getName());
    	std::cout << "Onnx Output \"" << network->getOutput(i)->getName() << "\": "
        		<< dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    //configure Builder
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1ULL << 30);			//max gpu memory 1 Giga
	builder->setFp16Mode(fp16);


	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

    IF_NOT (engine != nullptr) {
    	std::cout << "could not build engine" << std::endl;
        return nullptr;
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

bool cvInferenceBase::saveTrtModel(
		nvinfer1::ICudaEngine* engine,
		const std::string& fileName) {
	IF_NOT (engine) {
		return false;
	}

	//get model stream
	nvinfer1::IHostMemory *trtModelStream = engine->serialize();
	IF_NOT (trtModelStream) {
		return false;
	}

	//output trt model to file
    std::ofstream out(fileName, std::ios::out|std::ios::binary);
    if (!out) {
    	std::cout << "could not open plan output file: " << fileName << std::endl;
        return false;
    }

    //write to trt model file
    out.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    trtModelStream->destroy();
    out.close();

    //output input/output name to file
    std::string strNameFile = fileName + ".name";
    std::ofstream name_out(strNameFile, std::ios::out);
    if (!name_out) {
    	std::cout << "could not open input/output name file: " << strNameFile << std::endl;
        return false;
    }

    //output inputs name
    name_out << m_input_name.size() << std::endl;
    for (int i = m_input_name.size() - 1; i >= 0; i--) {
    	name_out << m_input_name[i] << std::endl;
    }

    //output outputs name
    name_out << m_output_name.size() << std::endl;
    for (int i = m_output_name.size() - 1; i >= 0; i--) {
    	name_out << m_output_name[i] << std::endl;
    }

    name_out.close();
	return true;
}


bool cvInferenceBase::readTrtModel(const std::string& fileName, std::shared_ptr<char>& engine_buffer, int& engine_buffer_size) {
    //check the file exist
	std::ifstream in(fileName.c_str(),std::ios::in | std::ios::binary);
    if (!in.is_open()){
        engine_buffer_size = 0;
        engine_buffer = nullptr;
        return false;
    }

    //open file and read to buffe
    in.seekg(0,std::ios::end);
    engine_buffer_size = in.tellg();
    in.seekg(0,std::ios::beg);
    engine_buffer.reset(new char[engine_buffer_size]);
    in.read(engine_buffer.get(),engine_buffer_size);
    in.close();

    //check the input/output names file
    std::string strNameFile = fileName + ".name";
	std::ifstream name_in(strNameFile.c_str(),std::ios::in);
    if (!name_in.is_open()){
        engine_buffer_size = 0;
        engine_buffer = nullptr;
        return false;
    }

    std::string str_size;
    int size;

    //inputs name
    getline(name_in, str_size);
    size = atoi(str_size.c_str());
    m_input_name.resize(size);
    while(size--) {
    	getline(name_in, m_input_name[size]);
    }

    //outputs name
    getline(name_in, str_size);
    size = atoi(str_size.c_str());
    m_output_name.resize(size);
    while(size--) {
    	getline(name_in, m_output_name[size]);
    }

    name_in.close();

    return true;
}
