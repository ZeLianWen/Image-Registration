#ifndef _match_h_
#define _match_h_

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

#include<vector>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;

const double dis_ratio = 0.9;//最近邻和次近邻距离比阈值
const float ransac_error = 1.5;//ransac算法误差阈值
const double FSC_ratio_low = 0.8;
const double FSC_ratio_up = 1;

enum DIS_CRIT{ Euclidean=0,COS};//距离度量准则

/*该函数根据最小均方误差原则，计算变换矩阵*/
static Mat LMS(const Mat&points_1, const Mat &points_2, string model, float &rmse);

/*该函数使用ransac算法删除错误匹配点对*/
Mat ransac(const vector<Point2f>&points_1, const vector<Point2f> &points_2, string model, float threshold, vector<bool> &inliers, float &rmse);

/*该函数使用FSC算法删除错误点对*/
Mat FSC(const vector<Point2f> &points1_low, const vector<Point2f> &points2_low,
	const vector<Point2f> &points1_up, const vector<Point2f> &points2_up, string model, float threshold, vector<bool> &inliers, float &rmse);

/*该函数形成两幅图像的棋盘图，并且融合这两幅棋盘图像*/
void mosaic_map(const Mat &image_1, const Mat &image_2, Mat &chessboard_1, Mat &chessboard_2, Mat &mosaic_image, int width);

/*该函数把两幅配准后的图像进行融合镶嵌*/
void image_fusion(const Mat &image_1, const Mat &image_2, const Mat T, Mat &fusion_image, Mat &mosaic_image);

/*该函数进行描述子的最近邻和次近邻匹配*/
void match_des(const Mat &des_1, const Mat &des_2, vector<vector<DMatch>> &dmatchs, DIS_CRIT dis_crite);

/*该函数删除错误匹配点对，并进行配准*/
Mat match(const Mat &image_1, const Mat &image_2, const vector<vector<DMatch>> &dmatchs, vector<KeyPoint> keys_1,
	vector<KeyPoint> keys_2, string model, vector<DMatch> &right_matchs, Mat &matched_line);

/*该函数计算参考图像中一个描述子和待配准图像中所有描述子之间的距离，并返回最近邻和次近邻，以及索引*/
static void min_dis_idx(const float *ptr_1, const Mat &des_2, int num_des2, int dims_des, float dis[2], int idx[2]);

#endif
