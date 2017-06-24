#include"display.h"

#include<opencv2\highgui\highgui.hpp>
#include<vector>
#include<sstream>


/***************************该函数把金字塔放在一副图像上(包络高斯金字塔和DOG金字塔)******************/
/*pyramid表示高斯金字塔或者DOG金字塔
 pyramid_image表示生成的金字塔图像
 nOctaveLayers表示每组中间层数，默认是3
 str表示是高斯金字塔还是DOG金字塔
*/
void  mosaic_pyramid(const vector<vector<Mat>> &pyramid, Mat &pyramid_image, int nOctaceLayers,string str)
{
	//获得每组金字塔图像大小
	vector<vector<int>> temp_size;
	for (auto beg = pyramid.cbegin(); beg != pyramid.cend(); ++beg)
	{
		vector<int> temp_vec;
		int rows = (*beg)[0].rows;
		int cols = (*beg)[0].cols;
		temp_vec.push_back(rows);
		temp_vec.push_back(cols);
		temp_size.push_back(temp_vec);
	}

	//计算最后生成的金字塔图像pyramid_image的大小
	int total_rows = 0, total_cols = 0;
	for (auto beg = temp_size.begin(); beg != temp_size.end(); ++beg)
	{
		total_rows = total_rows + (*beg)[0];//获取行大小
		if (beg == temp_size.begin()){
			if (str == string("高斯金字塔"))
				total_cols = (nOctaceLayers + 3)*((*beg)[1]);//获取列大小
			else if (str == string("DOG金字塔"))
				total_cols = (nOctaceLayers + 2)*((*beg)[1]);//获取列大小
		}
	}

	pyramid_image.create(total_rows, total_cols, CV_8UC1);
	int i = 0, accumulate_row = 0;
	for (auto beg = pyramid.cbegin(); beg != pyramid.cend(); ++beg, ++i)
	{
		int accumulate_col = 0;
		accumulate_row = accumulate_row + temp_size[i][0];
		for (auto it = (*beg).cbegin(); it != (*beg).cend(); ++it)
		{
			accumulate_col = accumulate_col + temp_size[i][1];
			Mat temp(pyramid_image, Rect(accumulate_col - temp_size[i][1], accumulate_row - temp_size[i][0], it->cols, it->rows));
			Mat temp_it;
			normalize(*it, temp_it, 0, 255, NORM_MINMAX, CV_32FC1);
			convertScaleAbs(temp_it, temp_it, 1, 0);
			temp_it.copyTo(temp);
		}
	}
}

/**************************该函数保存拼接后的高斯金字塔和DOG金字塔图像**************************/
/*gauss_pyr_1表示参考高斯金字塔
 dog_pyr_1表示参考DOG金字塔
 gauss_pyr_2表示待配准高斯金字塔
 dog_pyr_2表示待配准DOG金字塔 
 nOctaveLayers表示金字塔每组中间层数
 */
void write_mosaic_pyramid(const vector<vector<Mat>> &gauss_pyr_1, const vector<vector<Mat>> &dog_pyr_1,
	const vector<vector<Mat>> &gauss_pyr_2, const vector<vector<Mat>> &dog_pyr_2,int nOctaveLayers)
{

	//显示参考和待配准高斯金字塔图像
	Mat gauss_image_1, gauss_image_2;
	mosaic_pyramid(gauss_pyr_1, gauss_image_1,nOctaveLayers, string("高斯金字塔"));
	mosaic_pyramid(gauss_pyr_2, gauss_image_2,nOctaveLayers, string("高斯金字塔"));
	imwrite(".\\image_save\\参考图像高斯金字塔.jpg", gauss_image_1);
	imwrite(".\\image_save\\待配准图像高斯金字塔.jpg", gauss_image_2);

	//显示参考和待配准DOG金字塔图像
	Mat dog_image_1, dog_image_2;
	mosaic_pyramid(dog_pyr_1, dog_image_1, nOctaveLayers, string("DOG金字塔"));
	mosaic_pyramid(dog_pyr_2, dog_image_2, nOctaveLayers, string("DOG金字塔"));
	imwrite(".\\image_save\\参考图像DOG金字塔.jpg", dog_image_1);
	imwrite(".\\image_save\\待配准图像DOG金字塔.jpg", dog_image_2);
}




