#include"match.h"
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<algorithm>
#include<vector>
#include<cmath>

using namespace std;
using namespace cv;
using namespace gpu;

RNG rng(100);

/********该函数根据正确的匹配点对，计算出图像之间的变换关系********/
/*match1_xy表示参考图像特征点坐标集合,[M x 2]矩阵，M表示特征的个数
 match2_xy表示待配准图像特征点集合，[M x 2]矩阵，M表示特征点集合
 model表示变换类型，“相似变换”,"仿射变换","透视变换"
 rmse表示均方根误差
 返回值为计算得到的3 x 3矩阵参数
 */
static Mat LMS(const Mat&match1_xy, const Mat &match2_xy, string model, float &rmse)
{

	if (match1_xy.rows != match2_xy.rows)
		CV_Error(CV_StsBadArg, "LMS模块输入特征点对个数不一致！");

	if (!(model == string("affine") || model == string("similarity") || 
		model ==string("perspective")))
		CV_Error(CV_StsBadArg, "LMS模块图像变换类型输入错误！");

	const int N = match1_xy.rows;//特征点个数
	
	Mat match2_xy_trans, match1_xy_trans;//坐标转置
	transpose(match1_xy, match1_xy_trans);
	transpose(match2_xy, match2_xy_trans);

	Mat change = Mat::zeros(3, 3, CV_32FC1);

	//A*X=B,接下来部分仿射变换和透视变换一样,如果特征点个数是M，则A=[2*M,6]矩阵
	//A=[x1,y1,0,0,1,0;0,0,x1,y1,0,1;.....xn,yn,0,0,1,0;0,0,xn,yn,0,1]
	Mat A = Mat::zeros(2*N,6,CV_32FC1);
	for (int i = 0; i < N; ++i)
	{
		A.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);//x
		A.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);//y
		A.at<float>(2*i, 4) = 1.f;

		A.at<float>(2 * i + 1, 2) = match2_xy.at<float>(i, 0);
		A.at<float>(2 * i + 1, 3) = match2_xy.at<float>(i, 1);
		A.at<float>(2 * i+1, 5) = 1.f;
	}

	//如果特征点个数是M,那个B=[2*M,1]矩阵
	//B=[u1,v1,u2,v2,.....,un,vn]
	Mat B;
	B.create(2 * N, 1, CV_32FC1);
	for (int i = 0; i < N; ++i)
	{
		B.at<float>(2 * i, 0) = match1_xy.at<float>(i, 0);//x
		B.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1);//y
	}

	//如果是仿射变换
	if (model == string("affine"))
	{
		Vec6f values;
		solve(A, B, values, DECOMP_QR);
		change = (Mat_<float>(3,3)<<values(0), values(1), values(4),
			values(2), values(3), values(5),
			+0.0f, +0.0f, 1.0f);

		Mat temp_1 = change(Range(0, 2), Range(0, 2));//尺度和旋转量
		Mat temp_2 = change(Range(0, 2), Range(2, 3));//平移量
		
		Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
		Mat diff = match2_xy_change - match1_xy_trans;//求差
		pow(diff,2.f,diff);
		rmse = (float)sqrt(sum(diff)(0)*1.0/N);//sum输出是各个通道的和
	}
	//如果是透视变换
	else if (model == string("perspective"))
	{
		/*透视变换模型
		[u'*w,v'*w, w]'=[u,v,w]' = [a1, a2, a5;
		                            a3, a4, a6;
		                            a7, a8, 1] * [x, y, 1]'
		[u',v']'=[x,y,0,0,1,0,-u'x, -u'y;
		         0, 0, x, y, 0, 1, -v'x,-v'y] * [a1, a2, a3, a4, a5, a6, a7, a8]'
		即，Y = A*X     */

		Mat A2;
		A2.create(2 * N, 2, CV_32FC1);
		for (int i = 0; i < N; ++i)
		{
			A2.at<float>(2*i, 0) = match1_xy.at<float>(i, 0)*match2_xy.at<float>(i, 0)*(-1.f);
			A2.at<float>(2*i, 1) = match1_xy.at<float>(i, 0)*match2_xy.at<float>(i, 1)*(-1.f);

			A2.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1)*match2_xy.at<float>(i, 0)*(-1.f);
			A2.at<float>(2 * i + 1, 1) = match1_xy.at<float>(i, 1)*match2_xy.at<float>(i, 1)*(-1.f);
		}

		Mat A1;
		A1.create(2 * N, 8, CV_32FC1);
		A.copyTo(A1(Range::all(), Range(0, 6)));
		A2.copyTo(A1(Range::all(), Range(6, 8)));

		Mat values;
		solve(A1, B, values, DECOMP_QR);
		change.at<float>(0, 0) = values.at<float>(0);
		change.at<float>(0, 1) = values.at<float>(1);
		change.at<float>(0, 2) = values.at<float>(4);
		change.at<float>(1, 0) = values.at<float>(2);
		change.at<float>(1, 1) = values.at<float>(3);
		change.at<float>(1, 2) = values.at<float>(5);
		change.at<float>(2, 0) = values.at<float>(6);
		change.at<float>(2, 1) = values.at<float>(7);
		change.at<float>(2, 2) = 1.f;

		Mat temp1 = Mat::ones(1, N, CV_32FC1);
		Mat temp2;
		temp2.create(3, N, CV_32FC1);
		match2_xy_trans.copyTo(temp2(Range(0, 2), Range::all()));
		temp1.copyTo(temp2(Range(2, 3), Range::all()));

		Mat match2_xy_change = change * temp2;
		Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());
		float *temp_ptr = match2_xy_change.ptr<float>(2);
		for (int i = 0; i < N; ++i)
		{
			float div_temp = temp_ptr[i];
			match2_xy_change_12.at<float>(0, i) = match2_xy_change_12.at<float>(0, i) / div_temp;
			match2_xy_change_12.at<float>(1, i) = match2_xy_change_12.at<float>(1, i) / div_temp;
		}
		Mat diff = match2_xy_change_12 - match1_xy_trans;
		pow(diff, 2, diff);
		rmse = (float)sqrt(sum(diff)(0)*1.0/ N);//sum输出是各个通道的和
	}
	//如果是相似变换
	else if (model == string("similarity"))
	{
		/*[x, y, 1, 0;
		  y, -x, 0, 1] * [a, b, c, d]'=[u,v]*/

		Mat A3;
		A3.create(2 * N, 4, CV_32FC1);
		for (int i = 0; i < N; ++i)
		{
			A3.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);
			A3.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);
			A3.at<float>(2 * i, 2) = 1.f;
			A3.at<float>(2 * i, 3) = 0.f;

			A3.at<float>(2 * i+1, 0) = match2_xy.at<float>(i, 1);
			A3.at<float>(2 * i+1, 1) = match2_xy.at<float>(i, 0)*(-1.f);
			A3.at<float>(2 * i+1, 2) = 0.f;
			A3.at<float>(2 * i + 1, 3) = 1.f;
		}

		Vec4f values;
		solve(A3, B, values, DECOMP_QR);
		change = (Mat_<float>(3, 3) << values(0), values(1), values(2),
			values(1)*(-1.0f), values(0), values(3),
			+0.f, +0.f, 1.f);

		Mat temp_1 = change(Range(0, 2), Range(0, 2));//尺度和旋转量
		Mat temp_2 = change(Range(0, 2), Range(2, 3));//平移量

		Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
		Mat diff = match2_xy_change - match1_xy_trans;//求差
		pow(diff, 2, diff);
		rmse = (float)sqrt(sum(diff)(0)*1.0 / N);//sum输出是各个通道的和
	}

	return change;
}

/*********************该函数删除错误的匹配点对****************************/
/*points_1表示参考图像上匹配的特征点位置
 points_2表示待配准图像上的特征点位置集合
 model表示变换模型，“similarity”,"affine"，“perspective”
 threshold表示内点阈值
 inliers表示points_1和points_2中对应的点对是否是正确匹配，如果是，对应元素值为1，否组为0
 rmse表示最后所有正确匹配点对计算出来的误差
 返回一个3 x 3矩阵，表示待配准图像到参考图像的变换矩阵
 */
Mat ransac(const vector<Point2f> &points_1, const vector<Point2f> &points_2, string model, float threshold, vector<bool> &inliers, float &rmse)
{
	if (points_1.size() != points_2.size())
		CV_Error(CV_StsBadArg, "ransac模块输入特征点数量不一致！");

	if (!(model == string("affine") || model == string("similarity") ||
		model == string("perspective")))
		CV_Error(CV_StsBadArg, "ransac模块图像变换类型输入错误！");

	const size_t N = points_1.size();//特征点对数
	int n;
	size_t max_iteration, iterations;
	if (model == string("similarity")){
		n = 2;
		max_iteration = N*(N - 1) / 2;
	}
	else if (model == string("affine")){
		n = 3; 
		max_iteration = N*(N - 1)*(N - 2) / (2 * 3);
	}
	else if (model == string("perspective")){
		n = 4;
		max_iteration = N*(N - 1)*(N - 2) / (2 * 3);
	}

	if (max_iteration > 800)
		iterations = 800;
	else
		iterations = max_iteration;

	//取出保存在points_1和points_2中的点坐标，保存在Mat矩阵中，方便处理
	Mat arr_1, arr_2;//arr_1,和arr_2是一个[3 x N]的矩阵，每一列表示一个点坐标,第三行全是1
	arr_1.create(3, (int)N, CV_32FC1);
	arr_2.create(3, (int)N, CV_32FC1);
	float *p10 = arr_1.ptr<float>(0), *p11 = arr_1.ptr<float>(1),*p12 = arr_1.ptr<float>(2);
	float *p20 = arr_2.ptr<float>(0), *p21 = arr_2.ptr<float>(1), *p22 = arr_2.ptr<float>(2);
	for (size_t i = 0; i < N; ++i)
	{
		p10[i] = points_1[i].x;
		p11[i] = points_1[i].y;
		p12[i] = 1.f;

		p20[i] = points_2[i].x;
		p21[i] = points_2[i].y;
		p22[i] = 1.f;
	}

	Mat rand_mat;
	rand_mat.create(1, n, CV_32SC1);
	int *p = rand_mat.ptr<int>(0);
	Mat sub_arr1, sub_arr2;
	sub_arr1.create(n, 2, CV_32FC1);
	sub_arr2.create(n, 2, CV_32FC1);
	Mat T;//待配准图像到参考图像的变换矩阵
	int most_consensus_num = 0;//当前最优一致集个数初始化为0
	vector<bool> right;
	right.resize(N);
	inliers.resize(N);

	for (size_t i = 0; i < iterations;++i)
	{
		//随机选择n个不同的点对
		while (1)
		{
			randu(rand_mat, 0, double(N-1));//随机生成n个范围在[0,N-1]之间的数

			//保证这n个点坐标不相同
			if (n == 2 && p[0] != p[1] && 
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]))
				break;

			if (n == 3 && p[0] != p[1] && p[0] != p[2] && p[1] != p[2] &&
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
				(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
				(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
				(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]))
				break;

			if (n == 4 && p[0] != p[1] && p[0] != p[2] && p[0] != p[3] &&
				p[1] != p[2] && p[1] != p[3] && p[2] != p[3] &&
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
				(p10[p[0]] != p10[p[3]] || p11[p[0]] != p11[p[3]]) &&
				(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
				(p10[p[1]] != p10[p[3]] || p11[p[1]] != p11[p[3]]) &&
				(p10[p[2]] != p10[p[3]] || p11[p[2]] != p11[p[3]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
				(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
				(p20[p[0]] != p20[p[3]] || p21[p[0]] != p21[p[3]]) &&
				(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]) &&
				(p20[p[1]] != p20[p[3]] || p21[p[1]] != p21[p[3]]) &&
				(p20[p[2]] != p20[p[3]] || p21[p[2]] != p21[p[3]]))
				break;
		}

		//提取出n个点对
		for (int i = 0; i < n; ++i)
		{
			sub_arr1.at<float>(i, 0) = p10[p[i]];
			sub_arr1.at<float>(i, 1) = p11[p[i]];

			sub_arr2.at<float>(i, 0) = p20[p[i]];
			sub_arr2.at<float>(i, 1) = p21[p[i]];
		}

		//根据这n个点对，计算变换矩阵T
		T = LMS(sub_arr1, sub_arr2, model, rmse);
		
		int consensus_num = 0;//当前一致集个数
		if(model == string("perspective"))
		{

			Mat match2_xy_change = T * arr_2;
			Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());
			float *temp_ptr = match2_xy_change.ptr<float>(2);
			for (size_t i = 0; i < N; ++i)
			{
				float div_temp = temp_ptr[i];
				match2_xy_change_12.at<float>(0, (int)i) = match2_xy_change_12.at<float>(0, (int)i) / div_temp;
				match2_xy_change_12.at<float>(1, (int)i) = match2_xy_change_12.at<float>(1, (int)i) / div_temp;
			}
			Mat diff = match2_xy_change_12 - arr_1(Range(0,2),Range::all());
			pow(diff, 2, diff);

			//第一行和第二行求和
			
			Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());
			float *p_add = add.ptr<float>(0);
			for (size_t i = 0; i < N; ++i)
			{
				if (p_add[i] < threshold){//如果小于阈值
					right[i] = true;
					++consensus_num;
				}
				else
					right[i] = false;
			}
		}

		else if (model == string("affine") || model == string("similarity"))
		{
			Mat match2_xy_change = T * arr_2;
			Mat diff = match2_xy_change - arr_1;
			pow(diff, 2, diff);

			//第一行和第二行求和
			Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());
			float *p_add = add.ptr<float>(0);
			for (size_t i = 0; i < N; ++i)
			{
				if (p_add[i] < threshold){//如果小于阈值
					right[i] = true;
					++consensus_num;
				}
				else
					right[i] = false;
			}
		}

		//判断当前一致集是否是优于之前最优一致集
		if (consensus_num>most_consensus_num){
				most_consensus_num = consensus_num;
				for (size_t i = 0; i < N; ++i)
					inliers[i] = right[i];
			}//更新当前最优一致集个数

	}

	//删除重复点对
	for (size_t i = 0; i < N-1; ++i)
	{
		for (size_t j = i + 1; j < N; ++j)
		{
			if (inliers[i] && inliers[j])
			{
				if (p10[i] == p10[j] && p11[i] == p11[j] && p20[i] == p20[j] && p21[i] == p21[j])
				{
					inliers[j] = false;
					--most_consensus_num;
				}
			}
		}
	}

	//迭代结束，获得最优一致集合，根据这些最优一致集合计算出最终的变换关系T
	Mat consensus_arr1, consensus_arr2;
	consensus_arr1.create(most_consensus_num, 2, CV_32FC1);
	consensus_arr2.create(most_consensus_num, 2, CV_32FC1);
	int k = 0;
	for (size_t i = 0; i < N; ++i)
	{
		if (inliers[i])
		{
			consensus_arr1.at<float>(k, 0) = p10[i];
			consensus_arr1.at<float>(k, 1) = p11[i];

			consensus_arr2.at<float>(k, 0) = p20[i];
			consensus_arr2.at<float>(k, 1) = p21[i];
			++k;
		}
	}

	T = LMS(consensus_arr1, consensus_arr2, model, rmse);
	return T;
}

/*********************该函数基于FSC算法删除错误点,FSC算法优于RANSAC算法****************************/
/*points1_low表示使用低距离比阈值获得的参考图像特征点坐标
 points2_low表示使用低距离比阈值获得的待配准图像特征点坐标
 points1_up表示使用高距离比阈值获得的参考图像特征点坐标
 points2_up表示使用高距离比阈值获得的待配准图像特征点坐标，由此可见points1_up 和 points2_up包含的特征点个数
 多于points1_low 和 points2_low包含的特征点个数
 model表示变换模型，“similarity”,"affine"，“perspective”
 threshold表示内点阈值
 inliers表示points1_up和points2_up中对应的点对是否是正确匹配，如果是，对应元素值为1，否组为0
 rmse表示最后所有正确匹配点对计算出来的误差
 返回一个3 x 3矩阵，表示待配准图像到参考图像的变换矩阵
*/
Mat FSC(const vector<Point2f> &points1_low, const vector<Point2f> &points2_low, 
	const vector<Point2f> &points1_up,const vector<Point2f> &points2_up,string model, float threshold, vector<bool> &inliers, float &rmse)
{
	if (points1_low.size() != points2_low.size() && points1_up.size() != points2_up.size())
		CV_Error(CV_StsBadArg, "ransac模块输入特征点数量不一致！");

	if (!(model == string("affine") || model == string("similarity") ||
		model == string("perspective")))
		CV_Error(CV_StsBadArg, "ransac模块图像变换类型输入错误！");

	const size_t N = points1_low.size();//低距离比阈值时候特征点对数
	const size_t M = points1_up.size();//高距离比阈值时候特征点个数（M>N）
	int n;
	size_t max_iteration, iterations;
	if (model == string("similarity")){
		n = 2;
		max_iteration = N*(N - 1) / 2;
	}
	else if (model == string("affine")){
		n = 3;
		max_iteration = N*(N - 1)*(N - 2) / (2 * 3);
	}
	else if (model == string("perspective")){
		n = 4;
		max_iteration = N*(N - 1)*(N - 2) / (2 * 3);
	}

	if (max_iteration > 800)
		iterations = 800;
	else
		iterations = max_iteration;

	//取出保存在points1_low和points2_low中的点坐标，保存在Mat矩阵中，方便处理
	Mat arr1_low, arr2_low;//arr1_low,和arr2_low是一个[3 x N]的矩阵，每一列表示一个点坐标,第三行全是1
	arr1_low.create(3, (int)N, CV_32FC1);
	arr2_low.create(3, (int)N, CV_32FC1);
	float *p10 = arr1_low.ptr<float>(0), *p11 = arr1_low.ptr<float>(1), *p12 = arr1_low.ptr<float>(2);
	float *p20 = arr2_low.ptr<float>(0), *p21 = arr2_low.ptr<float>(1), *p22 = arr2_low.ptr<float>(2);
	for (size_t i = 0; i < N; ++i)
	{
		p10[i] = points1_low[i].x;
		p11[i] = points1_low[i].y;
		p12[i] = 1.f;

		p20[i] = points2_low[i].x;
		p21[i] = points2_low[i].y;
		p22[i] = 1.f;
	}
	//取出保存在points1_up和points2_up中的点坐标，保存在Mat矩阵中，方便处理
	Mat arr1_up, arr2_up;//arr1_up,和arr2_up是一个[3 x ]的矩阵，每一列表示一个点坐标,第三行全是1
	arr1_up.create(3, (int)M, CV_32FC1);
	arr2_up.create(3, (int)M, CV_32FC1);
	float *up10 = arr1_up.ptr<float>(0), *up11 = arr1_up.ptr<float>(1), *up12 = arr1_up.ptr<float>(2);
	float *up20 = arr2_up.ptr<float>(0), *up21 = arr2_up.ptr<float>(1), *up22 = arr2_up.ptr<float>(2);
	for (size_t i = 0; i < M; ++i)
	{
		up10[i] = points1_up[i].x;
		up11[i] = points1_up[i].y;
		up12[i] = 1.f;

		up20[i] = points2_up[i].x;
		up21[i] = points2_up[i].y;
		up22[i] = 1.f;
	}

	Mat rand_mat;
	rand_mat.create(1, n, CV_32SC1);
	int *p = rand_mat.ptr<int>(0);
	Mat sub_arr1, sub_arr2;
	sub_arr1.create(n, 2, CV_32FC1);
	sub_arr2.create(n, 2, CV_32FC1);
	Mat T;//待配准图像到参考图像的变换矩阵
	int most_consensus_num = 0;//当前最优一致集个数初始化为0
	vector<bool> right;
	right.resize(M);
	inliers.resize(M);

	for (size_t i = 0; i < iterations; ++i)
	{
		//随机选择n个不同的点对
		while (1)
		{
			randu(rand_mat, 0, double(N - 1));//随机生成n个范围在[0,N-1]之间的数

			//保证这n个点坐标不相同
			if (n == 2 && p[0] != p[1] &&
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]))
				break;

			if (n == 3 && p[0] != p[1] && p[0] != p[2] && p[1] != p[2] &&
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
				(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
				(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
				(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]))
				break;

			if (n == 4 && p[0] != p[1] && p[0] != p[2] && p[0] != p[3] &&
				p[1] != p[2] && p[1] != p[3] && p[2] != p[3] &&
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
				(p10[p[0]] != p10[p[3]] || p11[p[0]] != p11[p[3]]) &&
				(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
				(p10[p[1]] != p10[p[3]] || p11[p[1]] != p11[p[3]]) &&
				(p10[p[2]] != p10[p[3]] || p11[p[2]] != p11[p[3]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
				(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
				(p20[p[0]] != p20[p[3]] || p21[p[0]] != p21[p[3]]) &&
				(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]) &&
				(p20[p[1]] != p20[p[3]] || p21[p[1]] != p21[p[3]]) &&
				(p20[p[2]] != p20[p[3]] || p21[p[2]] != p21[p[3]]))
				break;
		}

		//提取出n个点对
		for (int i = 0; i < n; ++i)
		{
			sub_arr1.at<float>(i, 0) = p10[p[i]];
			sub_arr1.at<float>(i, 1) = p11[p[i]];

			sub_arr2.at<float>(i, 0) = p20[p[i]];
			sub_arr2.at<float>(i, 1) = p21[p[i]];
		}

		//根据这n个点对，计算变换矩阵T
		T = LMS(sub_arr1, sub_arr2, model, rmse);

		int consensus_num = 0;//当前一致集个数
		if (model == string("perspective"))
		{

			Mat match2_xy_change = T * arr2_up;
			Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());
			float *temp_ptr = match2_xy_change.ptr<float>(2);
			for (size_t i = 0; i < M; ++i)
			{
				float div_temp = temp_ptr[i];
				match2_xy_change_12.at<float>(0, (int)i) = match2_xy_change_12.at<float>(0, (int)i) / div_temp;
				match2_xy_change_12.at<float>(1, (int)i) = match2_xy_change_12.at<float>(1, (int)i) / div_temp;
			}
			Mat diff = match2_xy_change_12 - arr1_up(Range(0, 2), Range::all());
			pow(diff, 2, diff);

			//第一行和第二行求和

			Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());
			float *p_add = add.ptr<float>(0);
			for (size_t i = 0; i < M; ++i)
			{
				if (p_add[i] < threshold){//如果小于阈值
					right[i] = true;
					++consensus_num;
				}
				else
					right[i] = false;
			}
		}

		else if (model == string("affine") || model == string("similarity"))
		{
			Mat match2_xy_change = T * arr2_up;
			Mat diff = match2_xy_change - arr1_up;
			pow(diff, 2, diff);

			//第一行和第二行求和
			Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());
			float *p_add = add.ptr<float>(0);
			for (size_t i = 0; i < M; ++i)
			{
				if (p_add[i] < threshold){//如果小于阈值
					right[i] = true;
					++consensus_num;
				}
				else
					right[i] = false;
			}
		}

		//判断当前一致集是否是优于之前最优一致集
		if (consensus_num>most_consensus_num){
			most_consensus_num = consensus_num;
			for (size_t i = 0; i < M; ++i)
				inliers[i] = right[i];
		}//更新当前最优一致集个数

	}

	//删除重复点对
	for (size_t i = 0; i < M - 1; ++i)
	{
		for (size_t j = i + 1; j < M; ++j)
		{
			if (inliers[i] && inliers[j])
			{
				if ((up10[i] == up10[j] && up11[i] == up11[j] && up20[i] == up20[j] && up21[i] == up21[j]) ||
					(up10[i] == up10[j] && up11[i] == up11[j]) || (up20[i] == up20[j] && up21[i] == up21[j]))
				{
					inliers[j] = false;
					--most_consensus_num;
				}
			}
		}
	}

	//迭代结束，获得最优一致集合，根据这些最优一致集合计算出最终的变换关系T
	Mat consensus_arr1, consensus_arr2;
	consensus_arr1.create(most_consensus_num, 2, CV_32FC1);
	consensus_arr2.create(most_consensus_num, 2, CV_32FC1);
	int k = 0;
	for (size_t i = 0; i < M; ++i)
	{
		if (inliers[i])
		{
			consensus_arr1.at<float>(k, 0) = up10[i];
			consensus_arr1.at<float>(k, 1) = up11[i];

			consensus_arr2.at<float>(k, 0) = up20[i];
			consensus_arr2.at<float>(k, 1) = up21[i];
			++k;
		}
	}

	T = LMS(consensus_arr1, consensus_arr2, model, rmse);
	return T;
}


/********************该函数生成两幅图的棋盘网格图*************************/
/*image_1表示参考图像
 image_2表示配准后的待配准图像
 chessboard_1表示image_1的棋盘图像
 chessboard_2表示image_2的棋盘图像
 mosaic_image表示image_1和image_2的镶嵌图像
 width表示棋盘网格大小
 */
void mosaic_map(const Mat &image_1, const Mat &image_2, Mat &chessboard_1, Mat &chessboard_2, Mat &mosaic_image, int width)
{
	if (image_1.size != image_2.size)
		CV_Error(CV_StsBadArg, "mosaic_map模块输入两幅图大小必须一致！");

	//生成image_1的棋盘网格图
	chessboard_1 = image_1.clone();
	int rows_1 = chessboard_1.rows;
	int cols_1 = chessboard_1.cols;

	int row_grids_1 = cvFloor((double)rows_1 / width);//行方向网格个数
	int col_grids_1 = cvFloor((double)cols_1 / width);//列方向网格个数

	for (int i = 0; i < row_grids_1; i = i + 2){
		for (int j = 1; j < col_grids_1; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_1(range_y, range_x) = 0;
		}
	}

	for (int i = 1; i < row_grids_1; i = i + 2){
		for (int j = 0; j < col_grids_1; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_1(range_y, range_x) = 0;
		}
	}

	//生成image_2的棋盘网格图
	chessboard_2 = image_2.clone();
	int rows_2 = chessboard_2.rows;
	int cols_2 = chessboard_2.cols;

	int row_grids_2 = cvFloor((double)rows_2 / width);//行方向网格个数
	int col_grids_2 = cvFloor((double)cols_2 / width);//列方向网格个数

	for (int i = 0; i < row_grids_2; i = i + 2){
		for (int j = 0; j < col_grids_2; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_2(range_y, range_x) = 0;
		}
	}

	for (int i = 1; i < row_grids_2; i = i + 2){
		for (int j = 1; j < col_grids_2; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_2(range_y, range_x) = 0;
		}
	}

	mosaic_image = chessboard_1 + chessboard_2;
}


/***************该函数把配准后的图像进行融合*****************/
/*image_1表示参考图像
 image_2表示待配准图像
 T表示待配准图像到参考图像的变换关系矩阵
 fusion_image表示融合后的图像
 mosaic_image表示融合镶嵌后的图像
 */
void image_fusion(const Mat &image_1, const Mat &image_2, const Mat T, Mat &fusion_image, Mat &mosaic_image)
{
	if (!(image_1.depth() == CV_8U && image_2.depth() == CV_8U))
		CV_Error(CV_StsBadArg, "image_fusion模块仅支持uchar类型图像！");

	int rows_1 = image_1.rows, cols_1 = image_1.cols;
	int rows_2 = image_2.rows, cols_2 = image_2.cols;
	int channel_1 = image_1.channels();
	int channel_2 = image_2.channels();

	Mat image_1_temp, image_2_temp;
	if (channel_1 == 3 && channel_2 == 3){
		image_1_temp = image_1;
		image_2_temp = image_2;
	}
	else if (channel_1 == 1 && channel_2 == 3){
		image_1_temp = image_1;
		cvtColor(image_2, image_2_temp, CV_RGB2GRAY);
	}
	else if (channel_1 == 3 && channel_2 == 1){
		cvtColor(image_1, image_1_temp, CV_RGB2GRAY);
		image_2_temp = image_2;
	}
	else if (channel_1 == 1 && channel_2 == 1){
		image_1_temp = image_1;
		image_2_temp = image_2;
	}
		
	Mat T_temp = (Mat_<float>(3,3)<<1, 0, cols_1,
		                           0, 1, rows_1,
		                                0, 0, 1);
	Mat T_1 = T_temp*T;

	//对参考图像和待配准图像进行变换
	Mat trans_1,trans_2;//same type as image_2_temp 
	trans_1=Mat::zeros(3 * rows_1, 3 * cols_1, image_1_temp.type());
	image_1_temp.copyTo(trans_1(Range(rows_1,2*rows_1), Range(cols_1,2*cols_1)));
	warpPerspective(image_2_temp, trans_2, T_1, Size(3 * cols_1, 3 * rows_1), INTER_LINEAR);

	Mat trans = trans_2.clone();
	int nRows = rows_1;
	int nCols = cols_1*image_1_temp.channels();
	int len = nCols;
	for (int i = 0; i < nRows; ++i)
	{
		uchar *ptr_1 = trans_1.ptr<uchar>(i+rows_1);
		uchar *ptr = trans.ptr<uchar>(i+rows_1);
		for (int j = 0; j < nCols; ++j)
		{
			if (ptr[j + len] == 0)//对于非重合区域
				ptr[j + len] = ptr_1[j + len];
			else//对于重合区域
				ptr[j + len] = saturate_cast<uchar>(((float)ptr[j + len] + (float)ptr_1[j + len]) / 2);
		}
	}

	//删除多余的区域
	Mat left_up = T_1*(Mat_<float>(3, 1) << 0, 0, 1);//左上角
	Mat left_down = T_1*(Mat_<float>(3, 1) << 0, rows_2 - 1, 1);//左下角
	Mat right_up = T_1*(Mat_<float>(3, 1) << cols_2 - 1, 0, 1);//右上角
	Mat right_down = T_1*(Mat_<float>(3, 1) << cols_2 - 1, rows_2 - 1, 1);//右下角

	//对于透视变换，需要除以一个因子
	left_up = left_up / left_up.at<float>(2, 0);
	left_down = left_down / left_down.at<float>(2, 0);
	right_up = right_up / right_up.at<float>(2, 0);
	right_down = right_down / right_down.at<float>(2, 0);

	//计算x,y坐标的范围
	float temp_1 = min(left_up.at<float>(0, 0), left_down.at<float>(0, 0));
	float temp_2 = min(right_up.at<float>(0, 0), right_down.at<float>(0, 0));
	float min_x = min(temp_1, temp_2);

	temp_1 = max(left_up.at<float>(0, 0), left_down.at<float>(0, 0));
	temp_2 = max(right_up.at<float>(0, 0), right_down.at<float>(0, 0));
	float max_x = max(temp_1, temp_2);

	temp_1 = min(left_up.at<float>(1, 0), left_down.at<float>(1, 0));
	temp_2 = min(right_up.at<float>(1, 0), right_down.at<float>(1, 0));
	float min_y = min(temp_1, temp_2);

	temp_1 = max(left_up.at<float>(1, 0), left_down.at<float>(1, 0));
	temp_2 = max(right_up.at<float>(1, 0), right_down.at<float>(1, 0));
	float max_y = max(temp_1, temp_2);

	int X_min = max(cvFloor(min_x), 0);
	int X_max = min(cvCeil(max_x), 3 * cols_1-1);
	int Y_min = max(cvFloor(min_y), 0);
	int Y_max = min(cvCeil(max_y), 3 * rows_1-1);


	if (X_min>cols_1)
		X_min = cols_1;
	if (X_max<2 * cols_1-1)
		X_max = 2 * cols_1 - 1;
	if (Y_min>rows_1)
		Y_min = rows_1;
	if (Y_max<2 * rows_1-1)
		Y_max = 2 * rows_1 - 1;

	Range Y_range(Y_min, Y_max+1);
	Range X_range(X_min, X_max+1);
	fusion_image = trans(Y_range, X_range);
	Mat matched_image = trans_2(Y_range, X_range);

	Mat chessboard_1, chessboard_2;
	mosaic_map(trans_1(Y_range, X_range), trans_2(Y_range, X_range), 
		chessboard_1, chessboard_2, mosaic_image, 50);
	imwrite(".\\image_save\\参考图像棋盘图像.jpg", chessboard_1);
	imwrite(".\\image_save\\待配准图像棋盘图像.jpg", chessboard_2);
	imwrite(".\\image_save\\配准后的待配准图像.jpg", matched_image);
}

/*************该函数进行特征点的最近邻和次近邻匹配**************/
/*des_1表示参考图像的描述子矩阵，【M x 128】,M表示参考图像描述子个数
 des_2表示待配准图像的描述子矩阵,【N x 128】,N表示待配准图像描述子个数
 dmatchs表示匹配的特征点对
 dis_crite表示使用的距离准则，‘0’表示欧式距离，‘1’表示COS距离,使用欧式距离速度非常慢
 */
/*void match_des(const Mat &des_1, const Mat &des_2, vector<vector<DMatch>> &dmatchs,DIS_CRIT dis_crite)
{
	Mat trans_des_2;
	if (dis_crite==1)//如果是cos距离
		transpose(des_2, trans_des_2);//转置参考描述子矩阵

	int num_des_1 = des_1.rows;
	int num_des_2 = des_2.rows;
	dmatchs.clear();

	//对于参考图像上的每一点，和待配准图像进行匹配
	vector<DMatch> match;
	match.resize(2);
	Mat temp_dis;
	temp_dis.create(1, num_des_2, CV_32FC1);
	float *ptr_dis = temp_dis.ptr<float>(0);
	for (int i = 0; i < num_des_1; ++i)
	{
		if (dis_crite == 0)//如果是欧式距离
		{
			Mat temp = repeat(des_1.row(i), num_des_2, 1);
			Mat diff = temp - des_2;
			pow(diff, 2, diff);
			for (int j = 0; j < num_des_2; ++j)
				ptr_dis[j] = (float)sum(diff.row(j))[0];
			sqrt(temp_dis, temp_dis);
		}
		else if (dis_crite == 1)//如果是cos距离
		{
			temp_dis = des_1.row(i)*trans_des_2;
			for (int k = 0; k < num_des_2; ++k)
				ptr_dis[k] = acosf(ptr_dis[k]);//反余弦
		}
		
		Mat sort_dis, sort_idx;
		cv::sort(temp_dis, sort_dis, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);//升序排列
		cv::sortIdx(temp_dis, sort_idx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

		match[0].queryIdx = i;
		match[0].trainIdx = sort_idx.at<int>(0, 0);
		match[0].distance = sort_dis.at<float>(0, 0);

		match[1].queryIdx = i;
		match[1].trainIdx = sort_idx.at<int>(0, 1);
		match[1].distance = sort_dis.at<float>(0, 1);
		dmatchs.push_back(match);
	}
}*/

/******该函数计算参考图像一个描述子和参考图像所有描述子的欧式距离，并获得最近邻和次近邻距离，以及对应的索引*/
/*sub_des_1表示参考图像的一个描述子
 des_2表示待配准图像描述子
 num_des_2值待配准图像描述子个数
 dims_des指的是描述子维度
 dis保存最近邻和次近邻距离
 idx保存最近邻和次近邻索引
 */
static void min_dis_idx(const float *ptr_1, const Mat &des_2,int num_des2,int dims_des,float dis[2],int idx[2])
{
	float min_dis1=1000, min_dis2=2000;
	int min_idx1, min_idx2;

	for (int j = 0; j < num_des2; ++j)
	{
		const float *ptr_des_2 = des_2.ptr<float>(j);
		float cur_dis=0;
		for (int k = 0; k < dims_des; ++k)
		{
			float diff = ptr_1[k] - ptr_des_2[k];
			cur_dis+=diff*diff;
		}
		if (cur_dis < min_dis1){
			min_dis1 = cur_dis;
			min_idx1 = j;
		}
		else if (cur_dis>=min_dis1 && cur_dis < min_dis2){
			min_dis2 = cur_dis;
			min_idx2 = j;
		}
			
	}
	dis[0] = sqrt(min_dis1); dis[1] = sqrt(min_dis2);
	idx[0] = min_idx1; idx[1] = min_idx2;
}

/*加速版本的描述子匹配函数,该加速版本比前面版本速度提升了3倍*/
void match_des(const Mat &des_1, const Mat &des_2, vector<vector<DMatch>> &dmatchs, DIS_CRIT dis_crite)
{
	int num_des_1 = des_1.rows;
	int num_des_2 = des_2.rows;
	int dims_des = des_1.cols;

	vector<DMatch> match(2);

	//对于参考图像上的每一点，和待配准图像进行匹配
	if (dis_crite == 0)//欧几里得距离
	{
		for (int i = 0; i < num_des_1; ++i)//对于参考图像中的每个描述子
		{
			const float *ptr_des_1 = des_1.ptr<float>(i);
			float dis[2];
			int idx[2];
			min_dis_idx(ptr_des_1, des_2, num_des_2,dims_des,dis, idx);
			match[0].queryIdx = i;
			match[0].trainIdx = idx[0];
			match[0].distance = dis[0];

			match[1].queryIdx = i;
			match[1].trainIdx = idx[1];
			match[1].distance = dis[1];

			dmatchs.push_back(match);
		}
	}
	else if (dis_crite == 1)//cos距离
	{
		Mat trans_des2;
		transpose(des_2, trans_des2);
		double aa = (double)getTickCount();
		Mat mul_des=des_1*trans_des2;
		//gemm(des_1, des_2, 1, Mat(), 0, mul_des, GEMM_2_T);
		double time1 = ((double)getTickCount() - aa) / getTickFrequency();
		cout << "cos距离中矩阵乘法花费时间： " << time1 << "s" << endl;

		for (int i = 0; i < num_des_1; ++i)
		{ 
			float max_cos1 = -1000, max_cos2 = -2000;
			int max_idx1, max_idx2;

			float *ptr_1 = mul_des.ptr<float>(i);
			for (int j = 0; j < num_des_2; ++j)
			{
				float cur_cos = ptr_1[j];
				if (cur_cos>max_cos1){
					max_cos1 = cur_cos;
					max_idx1 = j;
				}
				else if (cur_cos<=max_cos1 && cur_cos > max_cos2){
					max_cos2 = cur_cos;
					max_idx2 = j;
				}
			}

			match[0].queryIdx = i;
			match[0].trainIdx = max_idx1;
			match[0].distance = acosf(max_cos1);

			match[1].queryIdx = i;
			match[1].trainIdx = max_idx2;
			match[1].distance = acosf(max_cos2);
			dmatchs.push_back(match);
		}
	}		
}

/*******************该函数删除错误匹配点对,并完成匹配************************/
/*image_1表示参考图像，
 image_2表示待配准图像
 dmatchs表示最近邻和次近邻匹配点对
 keys_1表示参考图像特征点集合
 keys_2表示待配准图像特征点集合
 model表示变换模型
 right_matchs表示参考图像和待配准图像正确匹配点对
 matched_line表示在参考图像和待配准图像上绘制连接线
 该函数返回变换模型参数
 */
Mat match(const Mat &image_1, const Mat &image_2, const vector<vector<DMatch>> &dmatchs, vector<KeyPoint> keys_1,
	       vector<KeyPoint> keys_2,string model,vector<DMatch> &right_matchs,Mat &matched_line)
{
	//获取初始匹配的关键点的位置
	vector<Point2f> points1_low, points2_low;
	vector<Point2f> points1_up, points2_up;

	vector<DMatch> low_matchs;
	vector<DMatch> up_matchs;
	for (size_t i = 0; i < dmatchs.size(); ++i)
	{
		double dis_1 = dmatchs[i][0].distance;
		double dis_2 = dmatchs[i][1].distance;
		//points1_low和points2_low仅仅保存满足距离比关系的特征点
		if ((dis_1 / dis_2) < FSC_ratio_low)
		{
			points1_low.push_back(keys_1[dmatchs[i][0].queryIdx].pt);
			points2_low.push_back(keys_2[dmatchs[i][0].trainIdx].pt);
			low_matchs.push_back(dmatchs[i][0]);//保存正确的dmatchs
		}
		//points1_up和points2_up不仅保存满足距离比关系的特征点，还保存不满足距离比关系的点
		if ((dis_1 / dis_2) <= FSC_ratio_up)
		{
			points1_up.push_back(keys_1[dmatchs[i][0].queryIdx].pt);
			points2_up.push_back(keys_2[dmatchs[i][0].trainIdx].pt);
			points1_up.push_back(keys_1[dmatchs[i][1].queryIdx].pt);
			points2_up.push_back(keys_2[dmatchs[i][1].trainIdx].pt);
			up_matchs.push_back(dmatchs[i][0]);
			up_matchs.push_back(dmatchs[i][1]);
		}
		
	}
	cout << "改进的FSC算法低距离比匹配特征点个数是： " << low_matchs.size() << endl;
	cout << "改进的FSC算法高距离比匹配特征点个数是： " << up_matchs.size() << endl;

	//使用FSC算法删除错误点对
	vector<bool> inliers;
	float rmse;
	Mat homography = FSC(points1_low, points2_low, points1_up, points2_up, model, ransac_error, inliers, rmse);

	//提取出处正确匹配点对
	auto itt = up_matchs.begin();
	for (auto it = inliers.begin(); it != inliers.end(); ++it, ++itt)
	{
		if (*it)//如果是正确匹配点对
		{
			right_matchs.push_back((*itt));
		}

	}
	cout << "使用改进的FSC删除错误点对发现正确匹配个数： " << right_matchs.size() << endl;
	cout << "误差rmse: " << rmse << endl;

	//绘制初始匹配点对连线图
	Mat initial_matched;
	drawMatches(image_1, keys_1, image_2, keys_2, low_matchs,
		initial_matched, Scalar(0, 255, 0), Scalar(255, 0, 0),
		vector<char>(), 2);
	imwrite(".\\image_save\\初始匹配点对.jpg", initial_matched);

	//绘制正确匹配点对连线图
	drawMatches(image_1, keys_1, image_2, keys_2, right_matchs,
		matched_line, Scalar(0, 255, 0), Scalar(255, 0, 0),
		vector<char>(), 2);
	imwrite(".\\image_save\\正确匹配点对.jpg", matched_line);

	//保存和显示检测到的特征点
	Mat keys_image_1, keys_image_2;
	drawKeypoints(image_1, keys_1, keys_image_1, Scalar(0, 255, 0), 0);
	drawKeypoints(image_2, keys_2, keys_image_2, Scalar(0, 255, 0), 0);
	imwrite(".\\image_save\\参考图像检测到的特征点.jpg", keys_image_1);
	imwrite(".\\image_save\\待配准图像检测到的.jpg", keys_image_2);

	return homography;
}
