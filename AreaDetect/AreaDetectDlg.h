

// AreaDetectDlg.h : 头文件
//
#pragma once
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "Halconcpp.h"
#include "afxcmn.h"
using namespace HalconCpp;
using namespace cv;
// CAreaDetectDlg 对话框
class CAreaDetectDlg : public CDialogEx
{
// 构造
public:
	CAreaDetectDlg(CWnd* pParent = NULL);	// 标准构造函数
	bool DisplayFullImage(HObject &ho_Image);
	bool FCN_Detect();
// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_AREADETECT_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;
	HTuple m_HWindowID;
	HTuple m_ImageWidth, m_ImageHeight;
	HObject m_Image;
	HObject m_ImgOutput;
	HObject m_RegOutput;
	Mat m_MatImage;
	double m_dDispImagePartRow0;
	double m_dDispImagePartCol0;
	double m_dDispImagePartRow1;
	double m_dDispImagePartCol1;
	bool bZoomFlag;


	int m_nImageListCurrentItem;
	bool m_bExchangeDispFlag;
	CString m_strFilename;
	std::vector<std::string> m_vfiles;
	std::vector <std::string> m_vfullfiles;
	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnLvnItemchangedList1(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnBnClickedButtonOpenimg();
	afx_msg void OnBnClickedButtonOpenfolder();
	afx_msg void OnBnClickedButtonDetect();
	afx_msg void OnBnClickedButtonExchangedip();
//	afx_msg void OnBnClickedCheck5();
//	afx_msg void OnEnChangeEdit2();
	CListCtrl m_ImageList;
//	afx_msg void OnLvnItemchangedList2(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnNMClickList2(NMHDR *pNMHDR, LRESULT *pResult);
//	afx_msg void OnLvnItemchangedList2(NMHDR *pNMHDR, LRESULT *pResult);
//	afx_msg void OnHdnItemchangedList2(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnLvnItemchangedList2(NMHDR *pNMHDR, LRESULT *pResult);
	CListCtrl m_RegionList;
	afx_msg void OnBnClickedCheckModify();
	afx_msg void OnBnClickedCheck3();
	afx_msg void OnBnClickedCheck4();
	double m_dEditRealLength;
	double m_dEditResolution;
	double m_dDistance;
	afx_msg void OnEnChangeEdit1();
	afx_msg void OnEnChangeEdit2();
	afx_msg void OnBnClickedButtonOpenfolder4();
	void UpdateAreaDisplayList();
	afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);
	bool ShowImage(HObject ho_Image);
	afx_msg void OnBnClickedButtonDisplayfullimg();
};
