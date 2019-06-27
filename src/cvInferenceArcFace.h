//
// Created by chenxq on 19-4-18.
//
#ifndef __CV_INFERENCE_ARCFACE_H__
#define __CV_INFERENCE_ARCFACE_H__

#include <iostream>

//opencv include
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

//tensorrt include
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

//module include
#include "cvInferenceBase.h"

class cvInferenceArcFace : public cvInferenceBase
{
public:
	cvInferenceArcFace() {}
	~cvInferenceArcFace() {}

public:
	virtual bool Init();

	//inference interface
	bool beforeInference(const cv::Mat& face);
	bool afterInference();

private:
	//initialize tensorrt enviroment
	virtual bool CreateTrtModel(const std::string& trt_model);

	bool InitRGBConst();

private:
	//char* or string are not allow to initialize here
	static const char* OUTPUT_BLOB_NAME;
	static const char* CAFFE_PROTO_FILE;
	static const char* CAFFE_MODEL_FILE;

	float m_floatRGB[256];
};

#endif //__CV_INFERENCE_ARCFACE_H__

