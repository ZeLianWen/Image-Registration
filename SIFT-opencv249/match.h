#ifndef _match_h_
#define _match_h_

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

#include<vector>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;

const double dis_ratio = 0.8;//最近邻和次近邻距离比阈值
const float ransac_error = 1.5;//ransac算法误差阈值

enum DIS_CRIT{ Euclidean=0,COS};//距离度量准则

/*该函数把两幅配准后的图像进行融合镶嵌*/
void image_fusion(const Mat &image_1, const Mat &image_2, const Mat T, Mat &fusion_image, Mat &mosaic_image, Mat &matched_image);

/*该函数进行描述子的最近邻和次近邻匹配*/
void match_des(const Mat &des_1, const Mat &des_2, vector<vector<DMatch>> &dmatchs, DIS_CRIT dis_crite);

/*该函数删除错误匹配点对，并进行配准*/
Mat match(const Mat &image_1, const Mat &image_2, const vector<vector<DMatch>> &dmatchs, vector<KeyPoint> keys_1,
	vector<KeyPoint> keys_2, string model, vector<DMatch> &right_matchs, Mat &matched_line);

#endif
