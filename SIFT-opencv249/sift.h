#ifndef _SIFT_H_
#define _SIFT_H_

#include<iostream>
#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

using namespace std;
using namespace cv;

/*************************定义常量*****************************/

//高斯核大小和标准差关系，size=2*(GAUSS_KERNEL_RATIO*sigma)+1,经常设置GAUSS_KERNEL_RATIO=2-3之间
const double GAUSS_KERNEL_RATIO = 3;

const int MAX_OCTAVES = 8;//金字塔最大组数

const float CONTR_THR = 0.04f;//默认是的对比度阈值(D(x))

const float CURV_THR = 10.0f;//关键点主曲率阈值

const float INIT_SIGMA = 0.5f;//输入图像的初始尺度

const int IMG_BORDER = 2;//图像边界忽略的宽度

const int MAX_INTERP_STEPS = 5;//关键点精确插值次数

const int ORI_HIST_BINS = 36;//计算特征点方向直方图的BINS个数

const float ORI_SIG_FCTR = 1.5f;//计算特征点主方向时候，高斯窗口的标准差因子

const float ORI_RADIUS = 3 * ORI_SIG_FCTR;//计算特征点主方向时，窗口半径因子

const float ORI_PEAK_RATIO = 0.8f;//计算特征点主方向时，直方图的峰值比

const int DESCR_WIDTH = 4;//描述子直方图的网格大小(4x4)

const int DESCR_HIST_BINS = 8;//每个网格中直方图角度方向的维度

const float DESCR_MAG_THR = 0.2f;//描述子幅度阈值

const float DESCR_SCL_FCTR = 3.0f;//计算描述子时，每个网格的大小因子




/************************sift类*******************************/
class MySift
{

public:
	//默认构造函数
	MySift(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
		double edgeThreshold = 10, double sigma = 1.6, bool double_size = true) :nfeatures(nfeatures),
		nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold),
		edgeThreshold(edgeThreshold), sigma(sigma), double_size(double_size){}

	//获得尺度空间每组中间层数
	int get_nOctave_layers() const { return nOctaveLayers; }

	//获得图像尺度是否扩大一倍
	bool get_double_size() const { return double_size; }

	//计算金字塔组数
	int num_octaves(const Mat &image) const;

	//生成高斯金字塔第一组，第一层图像
	void create_initial_image(const Mat &image, Mat &init_image) const;

	//创建高斯金字塔
	void build_gaussian_pyramid(const Mat &init_image, vector<vector<Mat>> &gauss_pyramid, int nOctaves) const;

	//创建高斯差分金字塔
	void build_dog_pyramid(vector<vector<Mat>> &dog_pyramid, const vector<vector<Mat>> &gauss_pyramid) const;

	//DOG金字塔特征点检测
	void find_scale_space_extrema(const vector<vector<Mat>> &dog_pyr, const vector<vector<Mat>> &gauss_pyr,
		vector<KeyPoint> &keypoints) const;

	//计算特征点的描述子
	void calc_descriptors(const vector<vector<Mat>> &dog_pyr, vector<KeyPoint> &keypoints,
		Mat &descriptors) const;

	//特征点检测
	void detect(const Mat &image, vector<vector<Mat>> &gauss_pyr, vector<vector<Mat>> &dog_pyr,
		vector<KeyPoint> &keypoints) const;

	//特征点描述
	void comput_des(const vector<vector<Mat>> &gauss_pyr, vector<KeyPoint> &keypoints, Mat &descriptors) const;


private:
	int nfeatures;//设定检测的特征点的个数值,如果此值设置为0，则不影响结果
	int nOctaveLayers;//每组金字塔中间层数
	double contrastThreshold;//对比度阈值（D(x)）
	double edgeThreshold;//特征点边缘曲率阈值
	double sigma;//高斯尺度空间初始层的尺度
	bool double_size;//是否上采样原始图像

};//注意类结束的分号

#endif