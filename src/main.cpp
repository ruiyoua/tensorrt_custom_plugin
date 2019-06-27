//
// Created by chenxq on 19-6-27.
// Contact: 76462242@qq.com
//

#include <iostream>

//opencv include
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cvInferenceArcFace.h"
#include "define.h"

int main(int argc, char** argv) {
	cvInferenceArcFace inf_face;

	cvInferenceBase::GlobalInit();

	inf_face.Init();

	cv::Mat img = cv::imread(std::string("image/0.jpg"));

	inf_face.beforeInference(img);
	inf_face.doInference();
	inf_face.afterInference();

	cvInferenceBase::GlobalFini();

	return 0;
}

