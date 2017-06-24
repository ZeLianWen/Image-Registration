#include"Sar_sift.h"
#include"match.h"

#include<opencv2\highgui\highgui.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#include<sstream>
#include<vector>
#include<fstream>

int main(int argc, char *argv[])
{
	//argv[0]=sar_sift.exe   argv[1]=参考图像，  argv[2]=待配准图像   argv[3]=变换类型
	if (argc < 4){
		cout << "输入参数数量不足！" << endl;
		return -1;
	}

	if (string(argv[3]) != string("similarity") && string(argv[3]) != string("affine")
		&& string(argv[3]) != string("perspective")){
		cout << "输入变换类型错误！" << endl;
		return -1;
	}
	string change_model = string(argv[3]);

	Mat image_1 = imread(argv[1], -1);
	Mat image_2 = imread(argv[2], -1);

	//string change_model = string("affine");
	//Mat image_1 = imread("F:\\class_file\\graduate_data\\图像配准\\sar_sift\\sar_sift_FSC_modified\\Debug\\20080618_3_1000x1000.bmp", -1);
	//Mat image_2 = imread("F:\\class_file\\graduate_data\\图像配准\\sar_sift\\sar_sift_FSC_modified\\Debug\\20090619_3_1000x1000.bmp", -1);

	if (!image_1.data || !image_2.data){
		cout << "图像数据加载失败！" << endl;
		return -1;
	}

	double total_beg = (double)getTickCount();//总时间开始
	//构建Sar_sift对象
	int nums_1 = image_1.rows*image_1.cols;
	int nums_2 = image_2.rows*image_2.cols;
	int nFeatures_1 = cvRound((double)nums_1*0.008);
	int nFeatures_2 = cvRound((double)nums_2*0.008);
	Sar_sift sar_sift_1(nFeatures_1, 8, 2, pow(2, 1.0 / 3.0), 0.8/5, 0.04);
	Sar_sift sar_sift_2(nFeatures_2, 8, 2, pow(2, 1.0 / 3.0), 0.8/5, 0.04);

	//参考图像特征点检测与描述
	vector<KeyPoint> keypoints_1;
	vector<Mat> sar_harris_fun_1, amplit_1, orient_1;
	double detect1_beg = (double)getTickCount();
	sar_sift_1.detect_keys(image_1, keypoints_1, sar_harris_fun_1, amplit_1, orient_1);
	double detect1_time = ((double)getTickCount() - detect1_beg) / getTickFrequency();
	cout << "参考图像特征点检测花费时间是： " << detect1_time << "s" << endl;
	cout << "参考图像检测特征点个数是： " << keypoints_1.size() << endl;

	double des1_beg = (double)getTickCount();
	Mat descriptors_1;
	sar_sift_1.comput_des(keypoints_1, amplit_1, orient_1, descriptors_1);
	double des1_time = ((double)getTickCount() - des1_beg) / getTickFrequency();
	cout << "参考图像特征点描述花费时间是： " << des1_time << "s" << endl;

	//待配准图像特征点检测与描述
	vector<KeyPoint> keypoints_2;
	vector<Mat> sar_harris_fun_2, amplit_2, orient_2;
	double detect2_beg = (double)getTickCount();
	sar_sift_2.detect_keys(image_2, keypoints_2, sar_harris_fun_2, amplit_2, orient_2);
	double detect2_time = ((double)getTickCount() - detect2_beg) / getTickFrequency();
	cout << "待配准图像特征点检测花费时间是： " << detect2_time << "s" << endl;
	cout << "待配准图像检测特征点个数是： " << keypoints_2.size() << endl;

	double des2_beg = (double)getTickCount();
	Mat descriptors_2;
	sar_sift_2.comput_des(keypoints_2, amplit_2, orient_2, descriptors_2);
	double des2_time = ((double)getTickCount() - des2_beg) / getTickFrequency();
	cout << "待配准图像特征点描述花费时间是： " << des2_time << "s" << endl;


	//描述子匹配
	double match_beg = (double)getTickCount();
	//Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher();
	Ptr<DescriptorMatcher> matcher = new BFMatcher(NORM_L2);
	vector<vector<DMatch>> dmatchs;
	match_des(descriptors_1, descriptors_2, dmatchs, COS);
	//matcher->knnMatch(descriptors_1, descriptors_2, dmatchs,2);

	vector<DMatch> right_matchs;
	Mat matched_line;
	Mat homography = match(image_1, image_2, dmatchs, keypoints_1, keypoints_2, change_model, right_matchs, matched_line);
	string str_1;
	stringstream ss;
	ss << right_matchs.size();
	ss >> str_1;
	namedWindow(string("正确匹配特征点连线图: ") + str_1, WINDOW_AUTOSIZE);
	imshow(string("正确匹配特征点连线图: ") + str_1, matched_line);
	double match_time = ((double)getTickCount() - match_beg) / getTickFrequency();
	cout << "特征点匹配阶段花费时间是： " << match_time << "s" << endl;
	cout << "待配准图像到参考图像的" << change_model << "变换矩阵是：" << endl;
	cout << homography << endl;

	double total_time = ((double)getTickCount() - total_beg) / getTickFrequency();
	cout << "总花费时间是： " << total_time << endl;

	//获得参考图像和待配准图像检测特征点在各层的分布个数
	vector<int> keys1_num(SAR_SIFT_LATERS), keys2_num(SAR_SIFT_LATERS);
	for (int i = 0; i < SAR_SIFT_LATERS; ++i)
	{
		keys1_num[i] = 0;//清零
		keys2_num[i] = 0;
	}
	for (size_t i = 0; i < keypoints_1.size(); ++i)
	{
		++keys1_num[keypoints_1[i].octave];
	}
	for (size_t i = 0; i < keypoints_2.size(); ++i)
	{
		++keys2_num[keypoints_2[i].octave];
	}

	//获得正确匹配点在尺度空间各层的分布
	vector<int> right_nums1(SAR_SIFT_LATERS), right_nums2(SAR_SIFT_LATERS);
	for (int i = 0; i < SAR_SIFT_LATERS; ++i)
	{
		right_nums1[i] = 0;//清零
		right_nums2[i] = 0;
	}
	for (size_t i = 0; i < right_matchs.size(); ++i)
	{
		++right_nums1[keypoints_1[right_matchs[i].queryIdx].octave];
		++right_nums2[keypoints_2[right_matchs[i].trainIdx].octave];
	}

	//把正确匹配点坐标和所在的组
	ofstream ofile(".\\position.txt");
	if (!ofile.is_open())
	{
		cout << "文件输出错误！" << endl;
		return -1;
	}
	ofile << "序号" << "   " << "参考坐标" << "   " <<"层号"<<"   "<<"强度"<<"    "
		<<"待配准坐标" <<"  "<<"层号"<<"  "<<"强度"<<endl;
	for (size_t i = 0; i < right_matchs.size(); ++i)
	{
		ofile << i << "->" << keypoints_1[right_matchs[i].queryIdx].pt << "    "
			<< keypoints_1[right_matchs[i].queryIdx].octave << "    "
			<< keypoints_1[right_matchs[i].queryIdx].response << "    "
			<< keypoints_2[right_matchs[i].trainIdx].pt << "    "
			<< keypoints_2[right_matchs[i].trainIdx].octave << "   "
			<< keypoints_2[right_matchs[i].trainIdx].response << endl;
	}

	ofile << "-------------------------------------------------------" << endl;
	ofile << "组号" << " " << "参考点数" << " " << "待配准点数" << " " << "参考正确数" << " " << "待配准正确数" << endl;
	for (int i = 0; i < SAR_SIFT_LATERS; ++i)
	{
		ofile << i<< "       " << keys1_num[i] << "        " << keys2_num[i] << "        " 
			<<right_nums1[i] << "        " << right_nums2[i] << endl;
	}
	
	//图像融合
	Mat fusion_image, mosaic_image, matched_image;
	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image);
	namedWindow("融合后的图像", WINDOW_AUTOSIZE);
	imshow("融合后的图像", fusion_image);
	imwrite(".\\image_save\\融合后的图像.jpg", fusion_image);
	namedWindow("融合镶嵌后的图像", WINDOW_AUTOSIZE);
	imshow("融合镶嵌后的图像", mosaic_image);
	imwrite(".\\image_save\\融合镶嵌后的图像.jpg", mosaic_image);



	waitKey(0);
	return 0;
}