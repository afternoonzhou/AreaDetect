#include "stdafx.h"
#include "algorithms.h"


void SelectMaxAreaRegion(HObject ho_Region, HObject *ho_MaxRegion)
{

	// Local iconic variables
	HObject  ho_ConnectedRegions1;

	// Local control variables
	HTuple  hv_Number, hv_Area4, hv_Row9, hv_Column9;
	HTuple  hv_Max, hv_Index1;

	//
	Connection(ho_Region, &ho_ConnectedRegions1);
	CountObj(ho_ConnectedRegions1, &hv_Number);
	if (0 != (hv_Number == 1))
	{
		CopyObj(ho_Region, &(*ho_MaxRegion), 1, 1);
		return;
	}
	AreaCenter(ho_ConnectedRegions1, &hv_Area4, &hv_Row9, &hv_Column9);
	TupleMax(hv_Area4, &hv_Max);
	TupleFindFirst(hv_Area4, hv_Max, &hv_Index1);
	SelectObj(ho_ConnectedRegions1, &(*ho_MaxRegion), hv_Index1 + 1);
	//
	return;
}

void Detect(HObject &ho_Image, HObject &ImgOutput)
{
	bool bIsInit = ho_Image.IsInitialized();
	if(!bIsInit)
		return;
	HTuple tCount;
	CountObj(ho_Image, &tCount);
	if (tCount == 0)
		return;


	HObject  ho_Image1, ho_Image2, ho_Image3;
	HObject  ho_ImageSub1, ho_ImageSub2, ho_ImageResult, ho_Region;
	HObject  ho_ImageResult1, ho_ImageResult2, ho_ImageResult3;
	HObject  ho_Region4, ho_ConnectedRegions2, ho_ObjectSelected;
	HObject  ho_RegionFillUp1, ho_ImageReduced, ho_ImageSub;
	HObject  ho_Region2, ho_ConnectedRegions, ho_RegionTrans;
	HObject  ho_c, ho_ImageMean, ho_RegionDilation1, ho_RegionDifference;
	HObject  ho_ImageReduced1, ho_ImageZoom, ho_DerivGauss, ho_ImageScaleMax1;
	HObject  ho_Region7, ho_RegionZoom, ho_ConnectedRegions3;
	HObject  ho_RegionDilation, ho_ResultRegion, ho_ObjectSelected1;
	HObject  ho_ImageReduced2, ho_Region8, ho_MaxRegion, ho_RegionClosing;
	HObject  ho_RegionFillUp2;

	// Local control variables
	HTuple  hv_ImageFiles, hv_j, hv_Width, hv_Height;
	HTuple  hv_Area, hv_Row, hv_Column, hv_Max, hv_Index, hv_Row1;
	HTuple  hv_Column1, hv_Phi, hv_Length1, hv_Length2, hv_MaxLength;
	HTuple  hv_resolution, hv_i, hv_UsedThreshold2, hv_Area1;
	HTuple  hv_Row2, hv_Column2, hv_Number, hv_Deviation, hv_Mean;
	HTuple  hv_Deviation1, hv_UsedThreshold3;


	HTuple end_val3 = (hv_ImageFiles.TupleLength()) - 1;
	HTuple step_val3 = 1;

		

	Decompose3(ho_Image, &ho_Image1, &ho_Image2, &ho_Image3);

	SubImage(ho_Image1, ho_Image2, &ho_ImageSub1, 1, 0);
	SubImage(ho_Image2, ho_Image3, &ho_ImageSub2, 1, 0);
	AddImage(ho_ImageSub1, ho_ImageSub2, &ho_ImageResult, 1, 0);
	Threshold(ho_ImageResult, &ho_Region, 12, 255);

	TransFromRgb(ho_Image1, ho_Image2, ho_Image3, &ho_ImageResult1, &ho_ImageResult2,
		&ho_ImageResult3, "hsv");

	GetImageSize(ho_Image1, &hv_Width, &hv_Height);
	//有效区域提取
	Threshold(ho_Image1, &ho_Region4, 20, 255);
	Connection(ho_Region4, &ho_ConnectedRegions2);
	AreaCenter(ho_ConnectedRegions2, &hv_Area, &hv_Row, &hv_Column);
	TupleMax(hv_Area, &hv_Max);
	TupleFindFirst(hv_Area, hv_Max, &hv_Index);
	SelectObj(ho_ConnectedRegions2, &ho_ObjectSelected, hv_Index + 1);
	FillUp(ho_ObjectSelected, &ho_RegionFillUp1);
	ReduceDomain(ho_Image, ho_RegionFillUp1, &ho_ImageReduced);

	//reduce_domain (ImageResult2, RegionFillUp1, ImageReduced3)
	//标尺提取
	SubImage(ho_Image1, ho_Image2, &ho_ImageSub, 1, 0);
	Threshold(ho_ImageSub, &ho_Region2, 40, 255);
	Connection(ho_Region2, &ho_ConnectedRegions);
	AreaCenter(ho_ConnectedRegions, &hv_Area, &hv_Row, &hv_Column);
	TupleMax(hv_Area, &hv_Max);
	TupleFindFirst(hv_Area, hv_Max, &hv_Index);
	SelectObj(ho_ConnectedRegions, &ho_ObjectSelected, hv_Index + 1);
	ShapeTrans(ho_ObjectSelected, &ho_RegionTrans, "rectangle2");
	SmallestRectangle2(ho_RegionTrans, &hv_Row1, &hv_Column1, &hv_Phi, &hv_Length1,
		&hv_Length2);
	TupleMax2(hv_Length1, hv_Length2, &hv_MaxLength);

	//分辨率：mm/pixel
	if (0 != (hv_MaxLength == 0))
	{
		hv_resolution = 0;
	}
	else
	{
		hv_resolution = (1.0 * 10) / hv_MaxLength;
	}



	if (HDevWindowStack::IsOpen())
		DispObj(ho_ImageReduced, HDevWindowStack::GetActive());
	//svm
	//gen_empty_obj (InsideRegion)
	//gen_empty_obj (OutsideRegion)
	//for i := 1 to 3 by 1
	//draw_region (PolygonRegion, 200000)
	//union2 (PolygonRegion, InsideRegion, InsideRegion)
	//endfor

	//for i := 1 to 3 by 1
	//draw_region (PolygonRegion, 200000)
	//union2 (PolygonRegion, OutsideRegion, OutsideRegion)
	//endfor

	//concat_obj (InsideRegion, OutsideRegion, Classes)
	//create_class_gmm (3, 2, 1, 'spherical', 'normalization', 10, 42, GMMHandle)
	//add_samples_image_class_gmm (Dsc0015, Classes, GMMHandle, 2.0)
	//train_class_gmm (GMMHandle, 100, 0.001, 'training', 0.0001, Centers, Iter)

	//clear_samples_class_gmm (GMMHandle)
	//classify_image_class_gmm (Dsc0015, ClassRegions, GMMHandle, 0.0001)
	//create_class_svm (3, 'rbf', 0.01, 0.0005, 1, 'novelty-detection', 'normalization', 3, SVMHandle)
	//add_samples_image_class_svm (ImageReduced, InsideRegion, SVMHandle)
	//add_samples_image_class_svm (ImageReduced, OutsideRegion, SVMHandle)
	//train_class_svm (SVMHandle, 0.01, 'default')


	//clear_samples_class_svm (SVMHandle)
	//classify_image_class_svm (ImageReduced, ClassRegions, SVMHandle)
	//region_to_mean (ClassRegions, Dsc0015, ImageClass)
	SubImage(ho_Image1, ho_Image3, &ho_c, 1, 0);
	MeanImage(ho_c, &ho_ImageMean, 19, 19);
	DilationCircle(ho_RegionTrans, &ho_RegionDilation1, 21.5);
	Difference(ho_RegionFillUp1, ho_RegionDilation1, &ho_RegionDifference);
	ReduceDomain(ho_ImageMean, ho_RegionDifference, &ho_ImageReduced1);
	ScaleImageMax(ho_ImageReduced1, &ho_ImageReduced1);
	MedianImage(ho_ImageReduced1, &ho_ImageReduced1, "circle", 25, "mirrored");

	ZoomImageSize(ho_ImageReduced1, &ho_ImageZoom, 512, 512, "constant");
	DerivateGauss(ho_ImageZoom, &ho_DerivGauss, 19, "laplace");

	ScaleImageMax(ho_DerivGauss, &ho_ImageScaleMax1);
	//BinaryThreshold(ho_ImageScaleMax1, &ho_Region7, "max_separability", "dark", &hv_UsedThreshold2);
	VarThreshold(ho_ImageScaleMax1, &ho_Region7, 200, 200, 0.5, 12, "dark");
	ZoomRegion(ho_Region7, &ho_RegionZoom, (1.0*hv_Width) / 512, (1.0*hv_Height) / 512);
	AreaCenter(ho_RegionZoom, &hv_Area1, &hv_Row2, &hv_Column2);
	Connection(ho_RegionZoom, &ho_ConnectedRegions3);
	DilationCircle(ho_ConnectedRegions3, &ho_RegionDilation, 301);

	AreaCenter(ho_RegionDilation, &hv_Area1, &hv_Row2, &hv_Column2);

	//regiongrowing_mean (ImageReduced3, Regions, Row2, Column2, 30, 70)

	GenEmptyObj(&ho_ResultRegion);
	CountObj(ho_ConnectedRegions3, &hv_Number);
	{
		HTuple end_val103 = hv_Number;
		HTuple step_val103 = 1;
		for (hv_i = 1; hv_i.Continue(end_val103, step_val103); hv_i += step_val103)
		{
			SelectObj(ho_RegionDilation, &ho_ObjectSelected1, hv_i);
			ReduceDomain(ho_ImageResult, ho_ObjectSelected1, &ho_ImageReduced2);

			PlaneDeviation(ho_ObjectSelected1, ho_ImageReduced2, &hv_Deviation);
			Intensity(ho_ObjectSelected1, ho_ImageReduced2, &hv_Mean, &hv_Deviation1);
			//fit_surface_first_order (ImageReduced2, ImageReduced2, 'huber', 3, 2, Alpha, Beta, Gamma)
			//gen_image_surface_first_order (ImageSurface, 'byte', Alpha, Beta, Gamma, 256, 256, Width, Height)
			//sub_image (ImageReduced2, ImageSurface, ImageSub1, 1, 0)

			//edges_sub_pix (ImageReduced2, Edges, 'canny', 3, 5, 15)
			//scale_image (ImageSub1, ImageScaled, 1, -30)
			Threshold(ho_ImageReduced2, &ho_Region8, hv_Mean + (hv_Deviation / 2), 255);
			BinaryThreshold(ho_ImageReduced2, &ho_Region8, "max_separability", "light",
				&hv_UsedThreshold3);
			SelectMaxAreaRegion(ho_Region8, &ho_MaxRegion);
			ClosingCircle(ho_MaxRegion, &ho_RegionClosing, 13.5);
			//shape_trans (MaxRegion, RegionTrans1, 'convex')
			ConcatObj(ho_ResultRegion, ho_RegionClosing, &ho_ResultRegion);
		}
	}

	FillUp(ho_ResultRegion, &ho_RegionFillUp2);

	HObject Output1;
	PaintRegion(ho_RegionFillUp2, ho_Image2, &Output1, 255, "fill");
	Compose3(ho_Image1, Output1, ho_Image3, &ImgOutput);
	
	//var_threshold (ImageScaleMax1, Region6, 15, 15, 0.2, 10, 'light')
	//scale_image_max (ImageReduced1, ImageScaleMax)
	//intensity (RegionDifference, ImageScaleMax, Mean, Deviation)
	//binary_threshold (ImageScaleMax, Region5, 'max_separability', 'light', UsedThreshold1)

	//regiongrowing (c, Regions1, 3, 3, 1, 20)
	//regiongrowing_mean (c, Regions, 2449, 2312, 12, 100)
	//watersheds (c, Basins, Watersheds)
	//watersheds_threshold (c, Basins1, 100)
	//sub_image (Image2, Image3, ImageSub2, 1, 0)
	//add_image (ImageSub1, ImageSub2, ImageResult, 0.5, 0)
	//median_separate (c, ImageSMedian, 225, 225, 'mirrored')


	//edges_image (ImageMean, ImaAmp, ImaDir, 'canny', 1, 'nms', 3, 8)
	//sub_image (ImageResult, ImageSMedian, ImageSub2, 1, 0)
	//binary_threshold (ImageSub2, Region3, 'max_separability', 'light', UsedThreshold)
	//closing_circle (Region3, RegionClosing, 3.5)
	//connection (RegionClosing, ConnectedRegions1)
	//select_shape (ConnectedRegions1, SelectedRegions, ['circularity','area'], 'and', [0.3275,39610.4], [1,100000])
	//fill_up (SelectedRegions, RegionFillUp)
	//threshold (Image1, Region, 50, 255)
	//var_threshold (Image3, Region1, 405, 405, 0.1, 18, 'dark')
}
Mat HObject2Mat(HObject Hobj)
{
	HTuple htCh = HTuple();
	HTuple cType;
	Mat Image;
	ConvertImageType(Hobj, &Hobj, "byte");
	CountChannels(Hobj, &htCh);
	HTuple wid;
	HTuple hgt;
	int W, H;
	if (htCh[0].I() == 1)
	{
		HTuple ptr;
		GetImagePointer1(Hobj, &ptr, &cType, &wid, &hgt);
		W = (Hlong)wid;
		H = (Hlong)hgt;
		Image.create(H, W, CV_8UC1);
		uchar* pdata = (uchar*)ptr[0].L();
		memcpy(Image.data, pdata, W*H);
	}
	else if (htCh[0].I() == 3)
	{
		HTuple ptrR, ptrG, ptrB;
		GetImagePointer3(Hobj, &ptrR, &ptrG, &ptrB, &cType, &wid, &hgt);
		W = (Hlong)wid;
		H = (Hlong)hgt;
		Image.create(H, W, CV_8UC3);
		vector<Mat> vecM(3);
		vecM[2].create(H, W, CV_8UC1);
		vecM[1].create(H, W, CV_8UC1);
		vecM[0].create(H, W, CV_8UC1);
		uchar* pr = (uchar*)ptrR[0].L();
		uchar* pg = (uchar*)ptrG[0].L();
		uchar* pb = (uchar*)ptrB[0].L();
		memcpy(vecM[2].data, pr, W*H);
		memcpy(vecM[1].data, pg, W*H);
		memcpy(vecM[0].data, pb, W*H);
		merge(vecM, Image);
	}
	return Image;
}


HObject Mat2HObject(Mat& image)
{
	HObject Hobj = HObject();
	int hgt = image.rows;
	int wid = image.cols;
	int i;
	//	CV_8UC3
	if (image.type() == CV_8UC3)
	{
		vector<Mat> imgchannel;
		split(image, imgchannel);
		Mat imgB = imgchannel[0];
		Mat imgG = imgchannel[1];
		Mat imgR = imgchannel[2];
		uchar* dataR = new uchar[hgt*wid];
		uchar* dataG = new uchar[hgt*wid];
		uchar* dataB = new uchar[hgt*wid];
		for (i = 0; i<hgt; i++)
		{
			memcpy(dataR + wid*i, imgR.data + imgR.step*i, wid);
			memcpy(dataG + wid*i, imgG.data + imgG.step*i, wid);
			memcpy(dataB + wid*i, imgB.data + imgB.step*i, wid);
		}
		GenImage3(&Hobj, "byte", wid, hgt, (Hlong)dataR, (Hlong)dataG, (Hlong)dataB);
		delete[]dataR;
		delete[]dataG;
		delete[]dataB;
		dataR = NULL;
		dataG = NULL;
		dataB = NULL;
	}
	//	CV_8UCU1
	else if (image.type() == CV_8UC1)
	{
		uchar* data = new uchar[hgt*wid];
		for (i = 0; i<hgt; i++)
			memcpy(data + wid*i, image.data + image.step*i, wid);
		GenImage1(&Hobj, "byte", wid, hgt, (Hlong)data);
		delete[] data;
		data = NULL;
	}
	return Hobj;
}

HObject DomainCrop(HObject &ImgSrc, int &nRowTop, int &nColLeft)
{
	HObject ho_Image1, ho_Image2, ho_Image3, ho_Region4, ho_ConnectedRegions2;
	HObject ho_ObjectSelected, ho_RegionFillUp1, ho_RegionTrans, ho_ImageReduced, ho_ImagePart;
	HTuple tRowTop, tColLeft;
	Decompose3(ImgSrc, &ho_Image1, &ho_Image2, &ho_Image3);
	//有效区域提取
	Threshold(ho_Image1, &ho_Region4, 20, 255);
	Connection(ho_Region4, &ho_ConnectedRegions2);
	SelectShapeStd(ho_ConnectedRegions2, &ho_ObjectSelected, "max_area", 70);
	FillUp(ho_ObjectSelected, &ho_RegionFillUp1);
	ShapeTrans(ho_RegionFillUp1, &ho_RegionTrans, "inner_circle");
	SmallestRectangle1(ho_RegionTrans, &tRowTop, &tColLeft, NULL, NULL);
	nRowTop = tRowTop.I();
	nColLeft = tColLeft.I();
	ReduceDomain(ImgSrc, ho_RegionTrans, &ho_ImageReduced);
	CropDomain(ho_ImageReduced, &ho_ImagePart);

	return ho_ImagePart;
}