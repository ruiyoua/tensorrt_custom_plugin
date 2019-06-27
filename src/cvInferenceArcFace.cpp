//
// Created by lumi on 19-4-18.
//
#include "cvInferenceArcFace.h"
#include <cuda_runtime_api.h>

const char* cvInferenceArcFace::OUTPUT_BLOB_NAME = "fc1";
const char* cvInferenceArcFace::CAFFE_PROTO_FILE = "./models/arcface.prototxt";
const char* cvInferenceArcFace::CAFFE_MODEL_FILE = "./models/arcface.caffemodel";


bool cvInferenceArcFace::Init(){
	m_pluginFactory = new CaffePluginFactory();
	static const std::string TRT_MODEL_FILENAME = std::string(TRT_DIRECTORY) + "/arcface.trt";
	CHECK(this->CreateTrtModel(TRT_MODEL_FILENAME));
	CHECK((this->InitTrtEngine(TRT_MODEL_FILENAME)));

	this->AllocBuffer();
	this->InitRGBConst();
	return true;
}

bool cvInferenceArcFace::CreateTrtModel(const std::string& trt_model) {
	std::ifstream check(trt_model.c_str(),std::ios::in | std::ios::binary);
	std::ifstream check_name((trt_model + ".name").c_str(), std::ios::in);
    if (check.is_open() && check_name.is_open()){
        std::cout << "ARC_FACE " << trt_model << " model file exist" << std::endl;
        return true;
    }

    std::cout << "ARC_FACE " << "create model file : " << trt_model << std::endl;
    m_output_name.push_back(OUTPUT_BLOB_NAME);

    nvinfer1::ICudaEngine* engine = cvInferenceBase::caffeToTRTModel(
			CAFFE_PROTO_FILE, CAFFE_MODEL_FILE,
			m_input_name, m_output_name, 1, m_pluginFactory, false);

	return cvInferenceBase::saveTrtModel(engine, trt_model);
}


bool cvInferenceArcFace::InitRGBConst() {
	//preprocess
	for (int i = 0; i < 256; i++) {
		m_floatRGB[i] = (i - 127.5) / 128.0;
	}

	return true;
}

//generate data for test
bool cvInferenceArcFace::beforeInference(const cv::Mat& face) {
	CHECK_EQ(m_input_dims.size(), 1);
	CHECK_EQ(m_input.size(), 1);
	//get input dims
	int in_channel = m_input_dims[0].d[0];
	int in_width = m_input_dims[0].d[1];
	int in_height = m_input_dims[0].d[2];

	CHECK_EQ(face.cols, in_width);
	CHECK_EQ(face.rows, in_height);

	//format transform BGRBGRBGR...BGRBGRBGR ------> RRRRR...RRRGGGGG...GGGBBBBB...BBB
	for (int i = 0; i < in_width * in_height; ++i) {
		//RRRRR...RRR
		m_input[0][i] = m_floatRGB[face.data[i * in_channel + 2]];
		//GGGGG...GGG
		m_input[0][i + in_width * in_height * 1] = m_floatRGB[face.data[i * in_channel + 1]];
		//BBBBB...BBB
		m_input[0][i + in_width * in_height * 2] = m_floatRGB[face.data[i * in_channel]];
	}

	return true;
}


bool cvInferenceArcFace::afterInference() {
	CHECK_EQ(m_output_dims.size(), 1);
	CHECK_EQ(m_output.size(), 1);

	//print feature
	float fSqareSum = 0; //(x1^2 + x2^2 + x3^2 + ... + x512^2)
	for (int i = 0; i < m_output_dims[0].d[0]; i++) {
		std::cout << m_output[0][i] << "\t";
		if ((i+1) % 10 == 0)
			std::cout << "\n";
	}

	return true;
}

