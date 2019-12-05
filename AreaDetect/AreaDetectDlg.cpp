
// AreaDetectDlg.cpp : 实现文件
//
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <io.h>
#include <time.h>
#include <iostream>
#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "stdafx.h"
#include "AreaDetect.h"
#include "AreaDetectDlg.h"
#include "afxdialogex.h"
#include <fstream>
#include <iostream>
#include "algorithms.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
using std::string;
using namespace cv;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;

void getFiles(string path, string exd, vector<string>& files, vector<string> &fullfiles);
void getFiles(string path, vector<string>& v_exd, vector<string>& files, vector<string> &fullfiles);
class VOSSegmentation {
public:
	VOSSegmentation() {}
	~VOSSegmentation() {}

	void V_ss_init(string model_path, string weight_path);
	vector<cv::Mat> V_ss_infer(cv::Mat image);
	float V_resnet_infer(cv::Mat image);
	void V_ss_close();

	const vector<caffe::Blob<float>*>& defect_detection();
	void image_preprocess(cv::Mat& origin_image);
private:
	//shared_ptr<Net<float> > caffe_net_;
	caffe::Net<float> *caffe_net_;
	vector<float> means_;
	int im_h_;
	int im_w_;
	int im_chan_;
	float* bottom_data_;
};
VOSSegmentation g_handle;
const vector<caffe::Blob<float>*>& VOSSegmentation::defect_detection()
{
	//struct timeval start, end;
	double start, end;
	start = clock();

	double timeuse = 0;
	//gettimeofday(&start, NULL);
	const vector<caffe::Blob<float>*>& result = caffe_net_->Forward(bottom_data_, NULL);
	//gettimeofday(&end, NULL);
	//timeuse = (end.tv_sec - start.tv_sec) * 1000000.f + (end.tv_usec - start.tv_usec);
	//timeuse /= 1000.f;
	end = clock();
	timeuse = end - start;
	std::cout << "foward time: " << timeuse << std::endl;
	return result;
}

void VOSSegmentation::image_preprocess(cv::Mat& origin_image)
{
	const int image_width = origin_image.cols;
	const int image_height = origin_image.rows;
	const int channel = origin_image.channels();

	std::vector<cv::Mat> channels(channel);
	cv::split(origin_image, channels);

	for (int c = 0; c < channel; ++c)
	{
		const uchar* dataPtr = channels[c].data;
		for (int h = 0; h < image_height; ++h) {
			for (int w = 0; w < image_width; ++w) {
				bottom_data_[(c * image_height + h) * image_width + w] = static_cast<float>(dataPtr[h * image_width + w]) - means_[c];
			}
		}
	}

	return;
}

void VOSSegmentation::V_ss_init(string model_path, string weight_path)
{
	//Caffe::set_mode(Caffe::GPU);
	//caffe_net_.reset(new Net<float>(model_path, caffe::TEST));
	caffe_net_ = new caffe::Net<float>(model_path, caffe::TEST);
	caffe_net_->CopyTrainedLayersFrom(weight_path);

	means_.push_back(104.f);
	means_.push_back(117.f);
	means_.push_back(123.f);
	im_w_ = 512;
	im_h_ = 512;
	im_chan_ = 3;
	bottom_data_ = new float[im_chan_ * im_h_ * im_w_];

	return;
}
float VOSSegmentation::V_resnet_infer(cv::Mat image)
{
	cv::Mat resImages;
	const int im_height = image.rows;
	const int im_width = image.cols;
	const float h_scale = (float)im_height / im_h_;
	const float w_scale = (float)im_width / im_w_;

	cv::Mat resize_image;
	cv::resize(image, resize_image, cv::Size(im_w_, im_h_), cv::INTER_LINEAR);
	image_preprocess(resize_image);
	const vector<caffe::Blob<float>*>& result = defect_detection();

	const float* prob_data = result[0]->cpu_data();
	//cv::Mat res_img(im_h_, im_w_, CV_8UC1, cv::Scalar::all(0));
	float MaxProb = 0;
	int nMaxIndex = 0;
	for (int i = 0; i < 3; ++i) {
		if (prob_data[i] >= MaxProb)
		{
			MaxProb = prob_data[i];
			nMaxIndex = i;
		}
	}
	return float(MaxProb + nMaxIndex);
}
vector<cv::Mat> VOSSegmentation::V_ss_infer(cv::Mat image)
{
	vector<cv::Mat> resImages;
	const int im_height = image.rows;
	const int im_width = image.cols;
	const float h_scale = (float)im_height / im_h_;
	const float w_scale = (float)im_width / im_w_;

	cv::Mat resize_image;
	cv::resize(image, resize_image, cv::Size(im_w_, im_h_), cv::INTER_LINEAR);
	image_preprocess(resize_image);
	const vector<caffe::Blob<float>*>& result = defect_detection();

	const float* prob_data = result[0]->cpu_data();
	cv::Mat res_img(im_h_, im_w_, CV_8UC1, cv::Scalar::all(0));
	for (int i = 0; i < im_h_ * im_w_; ++i) {
		if (prob_data[i] >= 0.001)
			res_img.data[i] = prob_data[i] * 255;
	}
	resImages.push_back(res_img);
	
	return resImages;
}

void VOSSegmentation::V_ss_close()
{
	if (bottom_data_)
	{
		delete[] bottom_data_;
		bottom_data_ = NULL;
	}
}

void getFiles(string path, string exd, vector<string>& files, vector<string> &fullfiles)
{
	//文件句柄
	long long  hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string pathName, exdName;
	//
	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			////			//如果是文件夹中仍有文件夹,迭代之
			////			//如果不是,加入列表
			////			// 不推荐使用，硬要使用的话，需要修改else 里面的语句
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				////			if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
				////			getFiles( pathName.assign(path).append("\\").append(fileinfo.name), exd, files );
			}
			else
			{
				////				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				fullfiles.push_back(pathName.assign(path).append("\\").append(fileinfo.name)); // 要得到绝对目录使用该语句
																							   ////					//如果使用
				files.push_back(fileinfo.name); // 只要得到文件名字使用该语句
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void getFiles(string path, vector<string>& v_exd, vector<string>& files, vector<string> &fullfiles)
{
	//文件句柄
	long long  hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	for (int i = 0; i < (int)v_exd.size(); i++)
	{

		string pathName, exdName;
		if (0 != strcmp(v_exd[i].c_str(), ""))
		{
			exdName = "\\*." + v_exd[i];
		}
		else
		{
			exdName = "\\*";
		}

		if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
		{
			do
			{
				////			//如果是文件夹中仍有文件夹,迭代之
				////			//如果不是,加入列表
				////			// 不推荐使用，硬要使用的话，需要修改else 里面的语句
				if ((fileinfo.attrib &  _A_SUBDIR))
				{
					////			if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					////			getFiles( pathName.assign(path).append("\\").append(fileinfo.name), exd, files );
				}
				else
				{
					////				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					fullfiles.push_back(pathName.assign(path).append("\\").append(fileinfo.name)); // 要得到绝对目录使用该语句
																								   ////					//如果使用
					files.push_back(fileinfo.name); // 只要得到文件名字使用该语句
				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}

	}
}
// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CAreaDetectDlg 对话框



CAreaDetectDlg::CAreaDetectDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_AREADETECT_DIALOG, pParent)
	, m_dEditRealLength(1.0)
	, m_dEditResolution(1.0)
	, m_dDistance(1.0)
	, bZoomFlag(true)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_bExchangeDispFlag = true;
	m_nImageListCurrentItem = -1;
}

void CAreaDetectDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LIST2, m_ImageList);
	DDX_Control(pDX, IDC_LIST1, m_RegionList);
}

BEGIN_MESSAGE_MAP(CAreaDetectDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_NOTIFY(LVN_ITEMCHANGED, IDC_LIST1, &CAreaDetectDlg::OnLvnItemchangedList1)
	ON_BN_CLICKED(IDC_BUTTON_OPENIMG, &CAreaDetectDlg::OnBnClickedButtonOpenimg)
	ON_BN_CLICKED(IDC_BUTTON_OPENFOLDER, &CAreaDetectDlg::OnBnClickedButtonOpenfolder)
	ON_BN_CLICKED(IDC_BUTTON_DETECT, &CAreaDetectDlg::OnBnClickedButtonDetect)
	ON_BN_CLICKED(IDC_BUTTON_EXCHANGEDIP, &CAreaDetectDlg::OnBnClickedButtonExchangedip)
	ON_NOTIFY(NM_CLICK, IDC_LIST2, &CAreaDetectDlg::OnNMClickList2)
ON_NOTIFY(LVN_ITEMCHANGED, IDC_LIST2, &CAreaDetectDlg::OnLvnItemchangedList2)
ON_BN_CLICKED(IDC_CHECK_MODIFY, &CAreaDetectDlg::OnBnClickedCheckModify)
ON_BN_CLICKED(IDC_CHECK3, &CAreaDetectDlg::OnBnClickedCheck3)
ON_BN_CLICKED(IDC_CHECK4, &CAreaDetectDlg::OnBnClickedCheck4)
ON_EN_CHANGE(IDC_EDIT1, &CAreaDetectDlg::OnEnChangeEdit1)
ON_EN_CHANGE(IDC_EDIT2, &CAreaDetectDlg::OnEnChangeEdit2)
ON_BN_CLICKED(IDC_BUTTON_OPENFOLDER4, &CAreaDetectDlg::OnBnClickedButtonOpenfolder4)
ON_WM_MOUSEWHEEL()
ON_BN_CLICKED(IDC_BUTTON_DISPLAYFULLIMG, &CAreaDetectDlg::OnBnClickedButtonDisplayfullimg)
END_MESSAGE_MAP()


// CAreaDetectDlg 消息处理程序

BOOL CAreaDetectDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	CString str;
	str.Format("%f", m_dEditRealLength);
	SetDlgItemText(IDC_EDIT1, str);
	str.Format("%f", m_dEditResolution);
	SetDlgItemText(IDC_EDIT2, str);
	str.Format("%f", m_dDistance);
	SetDlgItemText(IDC_EDIT3, str);
	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	m_ImageList.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_ONECLICKACTIVATE);
	m_ImageList.InsertColumn(0, _T("序号"), LVCFMT_CENTER, 36);
	m_ImageList.InsertColumn(1, _T("文件名"), LVCFMT_LEFT, 400);

	m_RegionList.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_ONECLICKACTIVATE);
	m_RegionList.InsertColumn(0, _T("序号"), LVCFMT_CENTER, 36);
	m_RegionList.InsertColumn(1, _T("面积"), LVCFMT_LEFT, 200);

	// TODO: 在此添加额外的初始化代码
	HTuple HWindowID;
	CRect Rect;
	CWnd * pWnd = GetDlgItem(IDC_STATIC);
	HWindowID = (Hlong)pWnd->m_hWnd;//获取窗口句柄
	pWnd->GetWindowRect(&Rect);
	OpenWindow(0, 0, Rect.Width(), Rect.Height(), HWindowID, "visible", "", &m_HWindowID);

	string model_path = "inference.prototxt";
	string weight_path = "LesionSurvey.caffemodel";
	//加载模型
	g_handle.V_ss_init(model_path, weight_path);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CAreaDetectDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CAreaDetectDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CAreaDetectDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CAreaDetectDlg::OnLvnItemchangedList1(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMLISTVIEW pNMLV = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	// TODO: 在此添加控件通知处理程序代码
	*pResult = 0;
}


void CAreaDetectDlg::OnBnClickedButtonOpenimg()
{
	// TODO: 在此添加控件通知处理程序代码
	CFileDialog dlg(TRUE, NULL, NULL,
		OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT,
		NULL);

	if (dlg.DoModal() == IDOK)
	{
		m_strFilename = dlg.GetPathName();
		
		//m_vfullfiles.clear();
		//m_vfullfiles.push_back(m_szFilename.GetBuffer());
		//::MessageBox(NULL,m_szFilename,NULL,MB_OK);
		//m_ImageList.DeleteAllItems();

		int nItemCount = m_ImageList.GetItemCount();
		CString strIndex;
		strIndex.Format("%d", nItemCount+1);
		m_nImageListCurrentItem = m_ImageList.InsertItem(nItemCount, strIndex);
		m_ImageList.SetItemText(m_nImageListCurrentItem, 1, m_strFilename);
		m_ImageList.SetItemState(m_nImageListCurrentItem, LVIS_SELECTED, LVIS_SELECTED | LVIS_FOCUSED);
		GetDlgItem(IDC_STATIC_IMGPATH)->SetWindowText("当前图片：" + m_ImageList.GetItemText(m_nImageListCurrentItem, 1));
		//halcon read_image函数读取jpg格式图片时获取的长款会与资源管理器读取的长宽颠倒
		//所以使用opencv imread读取后转换成hobject
		//ReadImage(&m_Image, m_strFilename.GetBuffer());
		Mat img;
		img = imread(m_strFilename.GetBuffer());
		//CString str;
		//str.Format("w:%d, h:%d", img.cols, img.rows);
		//::MessageBox(NULL, str, "提示", MB_OK); 
		m_Image = Mat2HObject(img);
		DisplayFullImage(m_Image);
		m_RegionList.DeleteAllItems();

	}

}


void CAreaDetectDlg::OnBnClickedButtonOpenfolder()
{
	// TODO: 在此添加控件通知处理程序代码
	static TCHAR strDirName[MAX_PATH];
	BROWSEINFO bi;
	CString szString = TEXT("选择文件夹");
	bi.hwndOwner = ::GetFocus();
	bi.pidlRoot = NULL;
	bi.pszDisplayName = strDirName;
	bi.lpszTitle = szString;
	bi.ulFlags = BIF_BROWSEFORCOMPUTER | BIF_DONTGOBELOWDOMAIN | BIF_RETURNONLYFSDIRS;
	bi.lpfn = NULL;
	bi.lParam = 0;
	bi.iImage = 0;

	LPITEMIDLIST pItemIDList = ::SHBrowseForFolder(&bi);
	if (pItemIDList == NULL)
	{
		return;
	}
	::SHGetPathFromIDList(pItemIDList, strDirName);
	m_vfiles.clear();
	m_vfullfiles.clear();
	vector<string> v_exd;
	v_exd.push_back("jpg");
	v_exd.push_back("jpeg");
	v_exd.push_back("bmp");
	v_exd.push_back("png");
	getFiles(strDirName, v_exd, m_vfiles, m_vfullfiles);
	if (m_vfiles.size() == 0)
		return;
	m_ImageList.DeleteAllItems();
	for (int i = 0; i < (int)m_vfullfiles.size(); i++)
	{
		int nItemCount = m_ImageList.GetItemCount();
		CString strIndex;
		strIndex.Format("%d", nItemCount + 1);
		int pos = m_ImageList.InsertItem(nItemCount, strIndex);
		m_ImageList.SetItemText(pos, 1, m_vfullfiles[i].c_str());
	}
	m_nImageListCurrentItem = 0;
	m_ImageList.SetItemState(m_nImageListCurrentItem, LVIS_SELECTED, LVIS_SELECTED | LVIS_FOCUSED);
	GetDlgItem(IDC_STATIC_IMGPATH)->SetWindowText("当前图片：" + m_ImageList.GetItemText(m_nImageListCurrentItem, 1));
	m_strFilename = m_vfullfiles[0].c_str();
	Mat img;
	img = imread(m_strFilename.GetBuffer());
	m_Image = Mat2HObject(img);
	m_ImgOutput = m_Image;
	//ReadImage(&m_Image, m_strFilename.GetBuffer());
	DisplayFullImage(m_Image);
	GenEmptyObj(&m_RegOutput);
	m_RegionList.DeleteAllItems();
}

bool CAreaDetectDlg::DisplayFullImage(HObject &ho_Image)
{
	//ReadImage(&ho_Image, "G:\\白菜坏死面积提取\\病斑大小测定\\DSC_0607.JPG");
	bool bIsInit = ho_Image.IsInitialized();
	if (!bIsInit)
		return false;
	HTuple tCount;
	CountObj(ho_Image, &tCount);
	if (tCount == 0)
		return false;
	GetImageSize(ho_Image, &m_ImageWidth, &m_ImageHeight);

	//设置窗口
	CRect Rect;
	CWnd * pWnd = GetDlgItem(IDC_STATIC);
	pWnd->GetWindowRect(&Rect);
	float fImage = m_ImageWidth.D() / m_ImageHeight.D();
	float fWindow = (float)Rect.Width() / Rect.Height();
	float Row0 = 0, Col0 = 0, Row1 = m_ImageWidth - 1, Col1 = m_ImageHeight - 1;
	if (fWindow > fImage)
	{
		float w = fWindow * m_ImageHeight;
		Row0 = 0;
		Col0 = -(w - m_ImageWidth) / 2;
		Row1 = m_ImageHeight - 1;
		Col1 = m_ImageWidth + (w - m_ImageWidth) / 2;
	}
	else
	{
		float h = m_ImageWidth / fWindow;
		Row0 = -(h - m_ImageHeight) / 2;
		Col0 = 0;
		Row1 = m_ImageHeight + (h - m_ImageHeight) / 2;
		Col1 = m_ImageWidth - 1;
	}

	m_dDispImagePartRow0 = Row0;
	m_dDispImagePartCol0 = Col0;
	m_dDispImagePartRow1 = Row1;
	m_dDispImagePartCol1 = Col1;
	ShowImage(ho_Image);
	//GetImagePointer1(ho_Image, NULL, NULL, &m_ImageWidth, &m_ImageHeight);
	//HTuple tMax;
	//TupleMax2(m_ImageWidth, m_ImageHeight, &tMax);
	/*m_dDispImagePartRow0 = 0;
	m_dDispImagePartCol0 = 0;
	m_dDispImagePartRow1 = m_ImageHeight;
	m_dDispImagePartCol1 = m_ImageWidth;
	SetPart(m_HWindowID, 0, 0, m_ImageHeight - 1, m_ImageWidth - 1);
	DispObj(ho_Image, m_HWindowID);*/
	//GetDlgItem(IDC_STATIC)->SetWindowText("hello");
	return true;
}


void  FindCorkDefect(const HObject &image, HTuple modelID, HTuple &phi1, HTuple &phi2, HObject & line1, HObject & line2, bool &flag)
{
	phi1 = HTuple();
	phi2 = HTuple();
	line1 = HObject();
	line2 = HObject();
	flag = true;

	HObject  ho_Regions, ho_ConnectedRegions, ho_SelectedRegions, ho_RegionTrans, ho_RegionFillUp, ho_ImageReduced;
	HObject  ho_RegionAffineTrans1, ho_p1, ho_Region1;
	HObject  ho_RegionTrans1, ho_Contours1, ho_ContoursSplit1;
	HObject  ho_RegionAffineTrans2, ho_p2, ho_Region2, ho_RegionTrans2;
	HObject  ho_Contours2, ho_ContoursSplit2, ho_UnionContours1, ho_UnionContours2;
	HObject  ho_SelectedContours1, ho_SelectedContours2;
	HTuple  hv_Area, hv_Row, hv_Column;
	HTuple  hv_Row1, hv_Column1, hv_Angle, hv_Score, hv_Length;
	HTuple  hv_HomMat2D1, hv_UsedThreshold, hv_RowBegin1, hv_ColBegin1;
	HTuple  hv_RowEnd1, hv_ColEnd1, hv_Nr, hv_Nc, hv_Dist, hv_row1;
	HTuple  hv_Min1, hv_N1, hv_HomMat2D2, hv_RowBegin2, hv_ColBegin2;
	HTuple  hv_RowEnd2, hv_ColEnd2, hv_row2, hv_Min2, hv_N2;

	HObject roi;
	GenRectangle1(&roi, 528.567, 733.9, 642.913, 957.9);
	AreaCenter(roi, &hv_Area, &hv_Row, &hv_Column);
	Threshold(image, &ho_Regions, 81, 255);
	Connection(ho_Regions, &ho_ConnectedRegions);
	SelectShapeStd(ho_ConnectedRegions, &ho_SelectedRegions, "max_area", 70);
	ShapeTrans(ho_SelectedRegions, &ho_RegionTrans, "convex");
	FillUp(ho_RegionTrans, &ho_RegionFillUp);
	ReduceDomain(image, ho_RegionFillUp, &ho_ImageReduced);

	FindNccModel(ho_ImageReduced, modelID, -0.39, 0.78, 0.5, 2, 0.5, "true", 0,
		&hv_Row1, &hv_Column1, &hv_Angle, &hv_Score);
	if (0 != (hv_Score > 0))
	{
		flag = true;
		TupleLength(hv_Score, &hv_Length);
		VectorAngleToRigid(hv_Row, hv_Column, 0, HTuple(hv_Row1[0]), HTuple(hv_Column1[0]),
			0, &hv_HomMat2D1);
		AffineTransRegion(roi, &ho_RegionAffineTrans1, hv_HomMat2D1, "nearest_neighbor");
		ReduceDomain(ho_ImageReduced, ho_RegionAffineTrans1, &ho_p1);
		BinaryThreshold(ho_p1, &ho_Region1, "max_separability", "dark", &hv_UsedThreshold);
		ShapeTrans(ho_Region1, &ho_RegionTrans1, "convex");
		GenContourRegionXld(ho_RegionTrans1, &ho_Contours1, "border");
		SegmentContoursXld(ho_Contours1, &ho_ContoursSplit1, "lines_circles", 5, 4, 2);
		UnionCollinearContoursXld(ho_ContoursSplit1, &ho_UnionContours1, 10, 1, 10, 0.3,
			"attr_keep");
		SelectContoursXld(ho_UnionContours1, &ho_SelectedContours1, "contour_length",
			100, 2000, -0.5, 0.5);
		FitLineContourXld(ho_SelectedContours1, "tukey", -1, 0, 5, 2, &hv_RowBegin1,
			&hv_ColBegin1, &hv_RowEnd1, &hv_ColEnd1, &hv_Nr, &hv_Nc, &hv_Dist);
		hv_row1 = hv_RowBegin1 + hv_RowEnd1;
		TupleMin(hv_row1, &hv_Min1);
		TupleFind(hv_row1, hv_Min1, &hv_N1);
		GenRegionLine(&line1, HTuple(hv_RowBegin1[hv_N1]), HTuple(hv_ColBegin1[hv_N1]),
			HTuple(hv_RowEnd1[hv_N1]), HTuple(hv_ColEnd1[hv_N1]));
		LineOrientation(HTuple(hv_RowBegin1[hv_N1]), HTuple(hv_ColBegin1[hv_N1]), HTuple(hv_RowEnd1[hv_N1]),
			HTuple(hv_ColEnd1[hv_N1]), &phi1);
		if (0 != (hv_Length == 2))
		{
			VectorAngleToRigid(hv_Row, hv_Column, 0, HTuple(hv_Row1[1]), HTuple(hv_Column1[1]),
				0, &hv_HomMat2D2);
			AffineTransRegion(roi, &ho_RegionAffineTrans2, hv_HomMat2D2, "nearest_neighbor");
			ReduceDomain(ho_ImageReduced, ho_RegionAffineTrans2, &ho_p2);
			BinaryThreshold(ho_p2, &ho_Region2, "max_separability", "dark", &hv_UsedThreshold);
			ShapeTrans(ho_Region2, &ho_RegionTrans2, "convex");
			GenContourRegionXld(ho_RegionTrans2, &ho_Contours2, "border");
			SegmentContoursXld(ho_Contours2, &ho_ContoursSplit2, "lines_circles", 5, 4,
				2);
			UnionCollinearContoursXld(ho_ContoursSplit2, &ho_UnionContours2, 10, 1, 10,
				0.3, "attr_keep");
			SelectContoursXld(ho_UnionContours2, &ho_SelectedContours2, "contour_length",
				100, 2000, -0.5, 0.5);
			FitLineContourXld(ho_SelectedContours2, "tukey", -1, 0, 5, 2, &hv_RowBegin2,
				&hv_ColBegin2, &hv_RowEnd2, &hv_ColEnd2, &hv_Nr, &hv_Nc, &hv_Dist);
			hv_row2 = hv_RowBegin2 + hv_RowEnd2;
			TupleMin(hv_row2, &hv_Min2);
			TupleFind(hv_row2, hv_Min2, &hv_N2);
			GenRegionLine(&line2, HTuple(hv_RowBegin2[hv_N2]), HTuple(hv_ColBegin2[hv_N2]),
				HTuple(hv_RowEnd2[hv_N2]), HTuple(hv_ColEnd2[hv_N2]));
			LineOrientation(HTuple(hv_RowBegin2[hv_N2]), HTuple(hv_ColBegin2[hv_N2]), HTuple(hv_RowEnd2[hv_N2]),
				HTuple(hv_ColEnd2[hv_N2]), &phi2);
		}
		else
		{
			phi2 = HTuple();
		}

	}
	else
	{
		flag = false;
	}
}


void CAreaDetectDlg::OnBnClickedButtonDetect()
{
	//Detect(m_Image, m_ImgOutput);
	//DisplayImage(m_ImgOutput);
	//HTuple tPointR, tPointG, tPointB;
	//GetImagePointer3(m_Image, &tPointR, &tPointG, &tPointB, NULL, NULL, NULL);
	//m_MatImage.create(512, 512, CV_8UC3);
	//vector<Mat> vecM(3);
	//Mat matR(()tPointR.L());	


	GetDlgItem(IDC_BUTTON_DETECT)->SetWindowText("检测中...");
	GetDlgItem(IDC_BUTTON_DETECT)->EnableWindow(FALSE);
	if (!FCN_Detect())
	{
		GetDlgItem(IDC_BUTTON_DETECT)->SetWindowText("病变区域提取");
		GetDlgItem(IDC_BUTTON_DETECT)->EnableWindow(TRUE);
		return;
	}
	GetDlgItem(IDC_BUTTON_DETECT)->SetWindowText("病变区域提取");
	GetDlgItem(IDC_BUTTON_DETECT)->EnableWindow(TRUE);

	if (!DisplayFullImage(m_ImgOutput))
	{
		return;
	}

	HObject RegConnection;
	HTuple tCount;
	Connection(m_RegOutput, &RegConnection);
	CountObj(RegConnection, &tCount);
	HTuple tArea;
	AreaCenter(RegConnection, &tArea, NULL, NULL);
	tArea = tArea * m_dEditResolution * m_dEditResolution;
	m_RegionList.DeleteAllItems();
	for (int i = 0; i < tArea.Length(); i++)
	{
		int nItemCount = m_RegionList.GetItemCount();
		CString strIndex, strArea;
		strIndex.Format("%d", nItemCount + 1);
		strArea.Format("%f", tArea[i].D());
		int pos = m_RegionList.InsertItem(nItemCount, strIndex);
		m_RegionList.SetItemText(pos, 1, strArea);

	}

	m_bExchangeDispFlag = false;

}




void CAreaDetectDlg::OnBnClickedButtonExchangedip()
{
	// TODO: 在此添加控件通知处理程序代码
	m_bExchangeDispFlag = !m_bExchangeDispFlag;
	if (m_bExchangeDispFlag)
	{
		
		bool bIsInit = m_Image.IsInitialized();
		if (!bIsInit)
			return;
		HTuple tCount;
		CountObj(m_Image, &tCount);
		if (tCount == 0)
			return;
		ShowImage(m_Image);
	}
	else
	{
		bool bIsInit = m_ImgOutput.IsInitialized();
		if (!bIsInit)
			return;
		HTuple tCount;
		CountObj(m_ImgOutput, &tCount);
		if (tCount == 0)
			return;
		ShowImage(m_ImgOutput);
	}

}



bool CAreaDetectDlg::FCN_Detect()
{
	bool bIsInit = m_Image.IsInitialized();
	if (!bIsInit)
		return false;
	HTuple tCount;
	CountObj(m_Image, &tCount);
	if (tCount == 0)
		return false;
	
	int nRowTop, nColLeft;
	HObject ImgCrop, ImgZoom;
	HTuple tWidth, tHeight;
	ImgCrop = DomainCrop(m_Image, nRowTop, nColLeft);
	GetImageSize(ImgCrop, &tWidth, &tHeight);
	ZoomImageSize(ImgCrop, &ImgZoom, 512, 512, "constant");
	m_MatImage = HObject2Mat(ImgZoom);
	vector<Mat> result_images = g_handle.V_ss_infer(m_MatImage);
	HObject Img, ImgOutput, ho_Image1, ho_Image2, ho_Image3, ho_RegionFillUp, Output1, RegZoom;
	Img = Mat2HObject(result_images[0]);
	Threshold(Img, &ImgOutput, 220, 255);
	FillUp(ImgOutput, &ho_RegionFillUp);
	ZoomRegion(ho_RegionFillUp, &RegZoom, tWidth / 512.0, tHeight / 512.0);

	MoveRegion(RegZoom, &m_RegOutput, nRowTop, nColLeft);
	Decompose3(m_Image, &ho_Image1, &ho_Image2, &ho_Image3);
	PaintRegion(m_RegOutput, ho_Image2, &Output1, 255, "fill");
	Compose3(ho_Image1, Output1, ho_Image3, &m_ImgOutput);

	return true;
}






void CAreaDetectDlg::OnNMClickList2(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMITEMACTIVATE pNMItemActivate = reinterpret_cast<LPNMITEMACTIVATE>(pNMHDR);
	// TODO: 在此添加控件通知处理程序代码
	//int n = m_ImageList.GetNextItem(-1, LVIS_SELECTED);
	//POSITION pos = m_ImageList.GetFirstSelectedItemPosition();
	//if (pos == NULL)
	//{
	//	if (m_nImageListCurrentItem != -1)
	//	{
	//		m_ImageList.SetItemState(m_nImageListCurrentItem, LVIS_SELECTED, LVIS_SELECTED | LVIS_FOCUSED);
	//	}
	//	return;
	//}
	//int nSelectedIndex = m_ImageList.GetNextSelectedItem(pos);
	//m_nImageListCurrentItem = nSelectedIndex;
	//GetDlgItem(IDC_STATIC_IMGPATH)->SetWindowText("当前图片：" + m_ImageList.GetItemText(m_nImageListCurrentItem, 1));

	/*CString str;
	str.Format("%d", n);
	::MessageBox(NULL, str, "提示", MB_OK);*/
	*pResult = 0;
}







void CAreaDetectDlg::OnLvnItemchangedList2(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMLISTVIEW pNMLV = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	// TODO: 在此添加控件通知处理程序代码

	if (LVIF_STATE == pNMLV->uChanged)
	{
		if (pNMLV->uNewState == (LVIS_FOCUSED | LVIS_SELECTED))
		{
			m_nImageListCurrentItem = m_ImageList.GetNextItem(-1, LVIS_SELECTED);
			m_strFilename = m_ImageList.GetItemText(m_nImageListCurrentItem, 1);
			GetDlgItem(IDC_STATIC_IMGPATH)->SetWindowText("当前图片：" + m_strFilename);
			Mat img;
			img = imread(m_strFilename.GetBuffer());
			m_Image = Mat2HObject(img);
			m_ImgOutput = m_Image;
			//ReadImage(&m_Image, m_strFilename.GetBuffer());
			DisplayFullImage(m_Image);
			m_RegionList.DeleteAllItems();
			GenEmptyObj(&m_RegOutput);
		}
	}
	*pResult = 0;
}


void CAreaDetectDlg::OnBnClickedCheckModify()//手动增加区域
{
	// TODO: 在此添加控件通知处理程序代码
	if (!m_RegOutput.IsInitialized() || !m_Image.IsInitialized())
	{
		((CButton *)GetDlgItem(IDC_CHECK_MODIFY))->SetCheck(0);
		return;
	}
	int nState = ((CButton *)GetDlgItem(IDC_CHECK_MODIFY))->GetCheck();
	if (nState == 1)
	{
		HObject Reg, ho_Image1, ho_Image2, ho_Image3, Output1;
		SetColor(m_HWindowID, "red");
		((CButton *)GetDlgItem(IDC_CHECK_MODIFY))->EnableWindow(FALSE);
		bZoomFlag = false;
		DrawRegion(&Reg, m_HWindowID);
		bZoomFlag = true;
		Union2(Reg, m_RegOutput, &m_RegOutput);
		Decompose3(m_Image, &ho_Image1, &ho_Image2, &ho_Image3);
		PaintRegion(m_RegOutput, ho_Image2, &Output1, 255, "fill");
		Compose3(ho_Image1, Output1, ho_Image3, &m_ImgOutput);

		if (!ShowImage(m_ImgOutput))
		{
			return;
		}
		UpdateAreaDisplayList();
		((CButton *)GetDlgItem(IDC_CHECK_MODIFY))->EnableWindow(TRUE);
		((CButton *)GetDlgItem(IDC_CHECK_MODIFY))->SetCheck(0);
	}

	//HTuple tR, tC;
	//DrawPoint(m_HWindowID, &tR, &tC);

}


void CAreaDetectDlg::OnBnClickedCheck3()//手动去除区域
{
	// TODO: Add your control notification handler code here

	if (!m_RegOutput.IsInitialized() || !m_Image.IsInitialized())
	{
		((CButton *)GetDlgItem(IDC_CHECK3))->SetCheck(0);
		return;
	}

	int nState = ((CButton *)GetDlgItem(IDC_CHECK3))->GetCheck();
	if (nState == 1)
	{
		HObject Reg, ho_Image1, ho_Image2, ho_Image3, Output1;
		SetColor(m_HWindowID, "red");
		((CButton *)GetDlgItem(IDC_CHECK3))->EnableWindow(FALSE);
		bZoomFlag = false;
		DrawRegion(&Reg, m_HWindowID);
		bZoomFlag = true;
		Difference(m_RegOutput, Reg, &m_RegOutput);
		Decompose3(m_Image, &ho_Image1, &ho_Image2, &ho_Image3);
		PaintRegion(m_RegOutput, ho_Image2, &Output1, 255, "fill");
		Compose3(ho_Image1, Output1, ho_Image3, &m_ImgOutput);

		if (!ShowImage(m_ImgOutput))
		{
			return;
		}
		UpdateAreaDisplayList();
		((CButton *)GetDlgItem(IDC_CHECK3))->EnableWindow(TRUE);
		((CButton *)GetDlgItem(IDC_CHECK3))->SetCheck(0);
	}

}
void CAreaDetectDlg::UpdateAreaDisplayList()
{
	HObject RegConnection;
	HTuple tCount, tRegOutputCount;
	if (!m_RegOutput.IsInitialized())
	{
		m_RegionList.DeleteAllItems();
		return;
	}
	CountObj(m_RegOutput, &tRegOutputCount);
	if (tRegOutputCount.I() == 0)
	{
		m_RegionList.DeleteAllItems();
		return;
	}
	Connection(m_RegOutput, &RegConnection);
	CountObj(RegConnection, &tCount);
	HTuple tArea;
	AreaCenter(RegConnection, &tArea, NULL, NULL);
	tArea = tArea * m_dEditResolution * m_dEditResolution;
	m_RegionList.DeleteAllItems();
	for (int i = 0; i < tArea.Length(); i++)
	{
		int nItemCount = m_RegionList.GetItemCount();
		CString strIndex, strArea;
		strIndex.Format("%d", nItemCount + 1);
		strArea.Format("%f", tArea[i].D());
		int pos = m_RegionList.InsertItem(nItemCount, strIndex);
		m_RegionList.SetItemText(pos, 1, strArea);

	}
}

void CAreaDetectDlg::OnBnClickedCheck4()//画标尺标定
{
	// TODO: Add your control notification handler code here

	CString str;
	GetDlgItemText(IDC_EDIT2, str);
	m_dEditRealLength = atof(str);
	int nState = ((CButton *)GetDlgItem(IDC_CHECK4))->GetCheck();
	if (nState == 1)
	{
		HObject Reg;
		HTuple hv_Row1, hv_Column1, hv_Row2, hv_Column2, hv_Distance;
		((CButton *)GetDlgItem(IDC_CHECK4))->EnableWindow(FALSE);
		SetColor(m_HWindowID, "blue");
		DrawLine(m_HWindowID, &hv_Row1, &hv_Column1, &hv_Row2, &hv_Column2);
		DistancePp(hv_Row1, hv_Column1, hv_Row2, hv_Column2, &hv_Distance);
		m_dDistance = hv_Distance.D();
		m_dEditResolution = m_dEditRealLength / m_dDistance;
		UpdateAreaDisplayList();

	}
	((CButton *)GetDlgItem(IDC_CHECK4))->EnableWindow(TRUE);
	((CButton *)GetDlgItem(IDC_CHECK4))->SetCheck(0);
	str.Format("%f", m_dEditResolution);
	SetDlgItemText(IDC_EDIT1, str);
	str.Format("%f", m_dDistance);
	SetDlgItemText(IDC_EDIT3, str);
}


void CAreaDetectDlg::OnEnChangeEdit1()//分辨率
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialogEx::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here

	//CString str;
	//str.Format("%f", m_dEditRealLength);
	//SetDlgItemText(IDC_EDIT1, str);
}


void CAreaDetectDlg::OnEnChangeEdit2()//真实距离
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialogEx::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
	CString str;
	GetDlgItemText(IDC_EDIT2, str);
	m_dEditRealLength = atof(str);
	m_dEditResolution = m_dEditRealLength / m_dDistance;
	UpdateAreaDisplayList();

}


void CAreaDetectDlg::OnBnClickedButtonOpenfolder4()//批量输出
{
	// TODO: Add your control notification handler code here
	std::string str = "resultFile.txt";
	ofstream outfile(str.c_str(), ios::app);
	outfile << m_strFilename.GetBuffer();
	outfile << " ";

	HObject RegConnection;
	HTuple tCount;
	Connection(m_RegOutput, &RegConnection);
	CountObj(RegConnection, &tCount);
	HTuple tArea;
	AreaCenter(RegConnection, &tArea, NULL, NULL);
	tArea = tArea * m_dEditResolution * m_dEditResolution;
	for (int i = 0; i < tArea.Length(); i++)
	{
		CString strIndex, strArea;
		//strIndex.Format("%d", nItemCount + 1);
		strArea.Format("%f", tArea[i].D());
		outfile << strArea.GetBuffer();
		outfile << " ";
		//int pos = m_RegionList.InsertItem(nItemCount, strIndex);
		//m_RegionList.SetItemText(pos, 1, strArea);

	}
	outfile << "\n";
	outfile.close();
}


bool CAreaDetectDlg::ShowImage(HObject ho_Image)
{
	bool bIsInit = ho_Image.IsInitialized();
	if (!bIsInit)
		return false;
	HTuple tCount;
	CountObj(ho_Image, &tCount);
	if (tCount == 0)
		return false;

	if (m_HWindowID != 0)
	{
		SetSystem("flush_graphic", "false");
		ClearWindow(m_HWindowID);
		//显示
		if (ho_Image.IsInitialized())
		{
			SetPart(m_HWindowID, m_dDispImagePartRow0, m_dDispImagePartCol0, m_dDispImagePartRow1 - 1, m_dDispImagePartCol1 - 1);
			DispObj(ho_Image, m_HWindowID);
		}

		SetSystem("flush_graphic", "true");
		HObject emptyObject;
		emptyObject.GenEmptyObj();
		DispObj(emptyObject, m_HWindowID);
	}
	return true;
}


BOOL CAreaDetectDlg::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	// TODO: Add your message handler code here and/or call default
	if(!bZoomFlag)
		return CDialogEx::OnMouseWheel(nFlags, zDelta, pt);

	CRect rtImage;
	GetDlgItem(IDC_STATIC)->GetWindowRect(&rtImage);
	if (rtImage.PtInRect(pt) && m_Image.IsInitialized())
	{
		Hlong  ImagePtX, ImagePtY;
		Hlong  Row0, Col0, Row1, Col1;
		double Scale = 0.1;

		if (zDelta<0)
		{
			ImagePtX = m_dDispImagePartCol0 + (pt.x - rtImage.left) / (rtImage.Width() - 1.0)*(m_dDispImagePartCol1 - m_dDispImagePartCol0);
			ImagePtY = m_dDispImagePartRow0 + (pt.y - rtImage.top) / (rtImage.Height() - 1.0)*(m_dDispImagePartRow1 - m_dDispImagePartRow0);
			Row0 = ImagePtY - 1 / (1 - Scale)*(ImagePtY - m_dDispImagePartRow0);
			Row1 = ImagePtY - 1 / (1 - Scale)*(ImagePtY - m_dDispImagePartRow1);
			Col0 = ImagePtX - 1 / (1 - Scale)*(ImagePtX - m_dDispImagePartCol0);
			Col1 = ImagePtX - 1 / (1 - Scale)*(ImagePtX - m_dDispImagePartCol1);

			m_dDispImagePartRow0 = Row0;
			m_dDispImagePartCol0 = Col0;
			m_dDispImagePartRow1 = Row1;
			m_dDispImagePartCol1 = Col1;
		}
		else
		{
			ImagePtX = m_dDispImagePartCol0 + (pt.x - rtImage.left) / (rtImage.Width() - 1.0)*(m_dDispImagePartCol1 - m_dDispImagePartCol0);
			ImagePtY = m_dDispImagePartRow0 + (pt.y - rtImage.top) / (rtImage.Height() - 1.0)*(m_dDispImagePartRow1 - m_dDispImagePartRow0);
			Row0 = ImagePtY - 1 / (1 + Scale)*(ImagePtY - m_dDispImagePartRow0);
			Row1 = ImagePtY - 1 / (1 + Scale)*(ImagePtY - m_dDispImagePartRow1);
			Col0 = ImagePtX - 1 / (1 + Scale)*(ImagePtX - m_dDispImagePartCol0);
			Col1 = ImagePtX - 1 / (1 + Scale)*(ImagePtX - m_dDispImagePartCol1);

			m_dDispImagePartRow0 = Row0;
			m_dDispImagePartCol0 = Col0;
			m_dDispImagePartRow1 = Row1;
			m_dDispImagePartCol1 = Col1;
		}
		if(m_bExchangeDispFlag)
			ShowImage(m_Image);
		else
			ShowImage(m_ImgOutput);
	}
	return CDialogEx::OnMouseWheel(nFlags, zDelta, pt);
}


void CAreaDetectDlg::OnBnClickedButtonDisplayfullimg()
{
	// TODO: Add your control notification handler code here
	bool bIsInit = m_Image.IsInitialized();
	if (!bIsInit)
		return;
	HTuple tCount;
	CountObj(m_Image, &tCount);
	if (tCount == 0)
		return;
	DisplayFullImage(m_Image);
}
