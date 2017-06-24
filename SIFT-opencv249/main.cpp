#include"sift.h"
#include"display.h"
#include"match.h"

#include<opencv2\highgui\highgui.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#include<fstream>
#include<stdlib.h>
#include<direct.h>


#pragma comment( lib,"IlmImf.lib" ) 
#pragma comment( lib,"libjasper.lib" )   
#pragma comment( lib,"libjpeg.lib" )   
#pragma comment( lib,"libpng.lib" )   
#pragma comment( lib,"libtiff.lib" )   
#pragma comment( lib, "zlib.lib") 
#pragma comment( lib, "opencv_ts249.lib") 

#pragma comment( lib, "vfw32.lib" )   
#pragma comment( lib, "comctl32.lib" ) 
#pragma comment( lib, "libcmt.lib" ) 


#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_features2d249.lib")
#pragma comment(lib,"opencv_flann249.lib")
#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")

int main(int argc,char *argv[])
{
	// string change_model;
	//if (argc <3 || argc>4){//输入参数个数必须是3或者4
	//	cout << "********输入参数错误,请输入三个或者四个正确参数！********" << endl;
	//	cout << "请按照下面顺序要求输入参数：" << endl;
	//	cout << "1. 可执行文件" << endl;
	//	cout << "2. 参考图像" << endl;
	//	cout << "3. 待配准图像" << endl;
	//	cout << "4. 变换类型" << endl;
	//	cout << "输入例子：" << endl;
	//	cout << "实例1： sift_2.exe school_3.jpg school_4.jpg similarity" << endl;
	//	cout << "实例2： sift_2.exe school_3.jpg school_4.jpg affine" << endl;
	//	cout << "实例3： sift_2.exe school_3.jpg school_4.jpg perspective" << endl;
	//	cout << "实例4： sift_2.exe school_3.jpg school_4.jpg" << endl;
	//	cout << "************************************************************" << endl;
	//	return -1;
	//}
	//else if (argc == 3)//如果是3个参数，默认选择透视变换模型
	//	change_model = string("perspective");
	//else if (argc == 4){
	//	change_model = string(argv[3]);//还可以是"similarity","perspective","affine"
	//	if (!(change_model == string("affine") || change_model == string("similarity") ||
	//		change_model == string("perspective")))
	//	{
	//		cout << "********变换类型输入错误！********" << endl;
	//		return -1;
	//	}
	//}

	////读入数据
	//Mat image_1, image_2;
	//image_1 = imread(argv[1], -1);
	//image_2 = imread(argv[2], -1);
	//if (!image_1.data || !image_2.data){
	//	cout << "图像数据加载失败！" << endl;
	//	return -1;
	//}

	/*Mat image_1 = imread("E:\\class_file\\graduate_data\\图像配准\\sift\\siftOpencv\\sift_static\\Debug\\ucsb1.jpg", -1);
	Mat image_2 = imread("E:\\class_file\\graduate_data\\图像配准\\sift\\siftOpencv\\sift_static\\Debug\\ucsb2.jpg", -1);*/

	//system("cd ..\\..\\");//返回上一级

	Mat image_1 = imread("..\\..\\set\\ucsb1.jpg", -1);
	Mat image_2 = imread("..\\..\\set\\ucsb2.jpg", -1);
	string change_model = "perspective";

	//创建文件夹保存图像
	char* newfile = ".\\image_save";
	_mkdir(newfile);

	//算法运行总时间开始计时
	double total_count_beg = (double)getTickCount();

	//类对象
	MySift sift_1(0, 3, 0.04, 10, 1.6, true);

	//参考图像特征点检测和描述
	vector<vector<Mat>> gauss_pyr_1, dog_pyr_1;
	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;
	double detect_1 = (double)getTickCount();
	sift_1.detect(image_1, gauss_pyr_1, dog_pyr_1, keypoints_1);
	double detect_time_1 = ((double)getTickCount() - detect_1) / getTickFrequency();
	cout << "参考图像特征点检测时间是： " << detect_time_1 << "s" << endl;
	cout << "参考图像检测特征点个数是： " << keypoints_1.size() << endl;

	double comput_1 = (double)getTickCount();
	sift_1.comput_des(gauss_pyr_1, keypoints_1, descriptors_1);
	double comput_time_1 = ((double)getTickCount() - comput_1) / getTickFrequency();
	cout << "参考图像特征点描述时间是： " << comput_time_1 << "s" << endl;


	//待配准图像特征点检测和描述
	vector<vector<Mat>> gauss_pyr_2, dog_pyr_2;
	vector<KeyPoint> keypoints_2;
	Mat descriptors_2;
	double detect_2 = (double)getTickCount();
	sift_1.detect(image_2, gauss_pyr_2, dog_pyr_2, keypoints_2);
	double detect_time_2 = ((double)getTickCount() - detect_2) / getTickFrequency();
	cout << "待配准图像特征点检测时间是： " << detect_time_2 << "s" << endl;
	cout << "待配准图像检测特征点个数是： " << keypoints_2.size() << endl;

	double comput_2 = (double)getTickCount();
	sift_1.comput_des(gauss_pyr_2, keypoints_2, descriptors_2);
	double comput_time_2 = ((double)getTickCount() - comput_2) / getTickFrequency();
	cout << "待配准特征点描述时间是： " << comput_time_2 << "s" << endl;

	//最近邻与次近邻距离比匹配
	double match_time = (double)getTickCount();
	Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher;
	//Ptr<DescriptorMatcher> matcher = new BFMatcher(NORM_L2);
	std::vector<vector<DMatch>> dmatchs;
	matcher->knnMatch(descriptors_1, descriptors_2, dmatchs, 2);
	//match_des(descriptors_1, descriptors_2, dmatchs, COS);

	Mat matched_lines;
	vector<DMatch> right_matchs;
	Mat homography = match(image_1, image_2, dmatchs, keypoints_1, keypoints_2, change_model,
		right_matchs,matched_lines);
	double match_time_2 = ((double)getTickCount() - match_time) / getTickFrequency();
	cout << "特征点匹配花费时间是： " << match_time_2 << "s" << endl;
	cout << change_model << "变换矩阵是：" << endl; 
	cout << homography << endl;

	//把正确匹配点坐标写入文件中
	ofstream ofile;
	ofile.open(".\\position.txt");
	for (size_t i = 0; i < right_matchs.size(); ++i)
	{
		ofile << keypoints_1[right_matchs[i].queryIdx].pt << "   "
			<< keypoints_2[right_matchs[i].trainIdx].pt << endl;
	}

	//图像融合
	double fusion_beg = (double)getTickCount();
	Mat fusion_image, mosaic_image, regist_image;
	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image, regist_image);
	imwrite(".\\image_save\\融合后的图像.jpg", fusion_image);
	imwrite(".\\image_save\\融合后的镶嵌图像.jpg", mosaic_image);
	imwrite(".\\image_save\\配准后的待配准图像.jpg", regist_image);
	double fusion_time = ((double)getTickCount() - fusion_beg) / getTickFrequency();
	cout << "图像融合花费时间是： " << fusion_time << "s" << endl;

	double total_time = ((double)getTickCount() - total_count_beg) / getTickFrequency();
	cout << "总花费时间是： " << total_time << "s" << endl;

	//显示匹配结果
	namedWindow("融合后的图像", WINDOW_AUTOSIZE);
	imshow("融合后的图像", fusion_image);
	namedWindow("融合镶嵌图像", WINDOW_AUTOSIZE);
	imshow("融合镶嵌图像", mosaic_image);
	stringstream s_2;
	string numstring_2, windowName;
	s_2 << right_matchs.size();
	s_2 >> numstring_2;
	windowName = string("正确点匹配连线图: ") + numstring_2;
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, matched_lines);

	//保存金字塔拼接好的金字塔图像
	//int nOctaveLayers = sift_1.get_nOctave_layers();
	//write_mosaic_pyramid(gauss_pyr_1, dog_pyr_1, gauss_pyr_2, dog_pyr_2, nOctaveLayers);

	waitKey(0);
	return 0;
}