#pragma once
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "Halconcpp.h"
#include <vector>
using namespace HalconCpp;
using namespace cv;
using namespace std;

void Detect(HObject &ho_Image, HObject &ImgOutput);
void SelectMaxAreaRegion(HObject ho_Region, HObject *ho_MaxRegion);
Mat HObject2Mat(HObject Hobj);
HObject Mat2HObject(Mat& image);
HObject DomainCrop(HObject &ImgSrc, int &nRowTop, int &nColLeft);