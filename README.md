
I'm terribly sorry, there are errors of the description in Section-III(B) of the paper “Remote Sensing Image Registration with Modified SIFT and Enhanced Feature Matching”.  We have uploaded the errors in document named 《 revised of the PSO-SIFT》.I'm sorry to have affected your reading.

我们已经出版的文章《Remote Sensing Image Registration with Modified SIFT and Enhanced Feature Matching》在第三部分（B）存在一些描述性错误，可能会给你的阅读带来麻烦，因此我们上传了出错的地方，并给出了正确的描述方法，文档名字为“revised of the PSO-SIFT”。CSDN 博主对已经翻译了该论文：https://blog.csdn.net/qq_21685903/article/details/103623053?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task


# Image-Registration
Image registration algorithm. Includes SIFT, SAR-SIFT,PSO-SIFT.If you are satisfied with our code,please click the star button(如果我们的代码给你的工作或者学习带来了帮助，希望能够收到你的star).

Author information：
Key Laboratory of Intelligent Perception and Image Understanding of Ministry of Education, International Research Center for Intelligent Perception and Computation, Joint International Research　Laboratory of Intelligent Perception and Computation, Xidian University, Xi’an, Shaanxi Province, 710071, China(email:zelianwen@foxmail.com).

Algorithm description：

1. SIFT(Scale-invariant feature transform). Reference article: David G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints"[2004].
2. SAR-SIFT. Reference article: Flora Dellinger, Julie Delon, Yann Gousseau, Julien Michel, and Florence Tupin, "SAR-SIFT: A SIFT-Like Algorithm for SAR Images"[2015].
3. PSO-SIFT. Reference article: "Remote Sensing Image Registration with Modified SIFT and Enhanced Feature Matching"[2016].If you do use it, please cite:
Ma W, Wen Z, Wu Y, et al. Remote Sensing Image Registration With Modified SIFT and Enhanced Feature Matching[J]. IEEE Geoscience and Remote Sensing Letters, 2017, 14(1): 3-7.
4. 基于点特征的遥感图像配准方法研究(Harris_Anisotropic scale space)，闻泽联。

SAR-SIFT-FSC-modified-opencv249是基于c++和opencv-2-4-9实现的SAR-SIFT算法，对于opencv比较熟悉的用户可以直接参考该代码，如果不熟悉opencv，可以参考matlab版本。

SIFT-opencv249是基于c++和opencv-2-4-9实现的原始SIFT算法，对于opencv比较熟悉的用户可以直接参考该代码，如果不熟悉opencv，可以参考matlab版本。

对于上面的opencv版本，我们仅仅提供了源文件，并没有提供编译脚本，对于windows下的用户只需要在IDE下加入配置开发环境即可；对于linux用户，可能需要自己编写编译脚本，我们推荐使用cmake。需要说明的是，SIFT-opencv249参考了opencv中sift实现代码，但是我们修改了其中的大部分代码，如高斯尺度空间生产、特征点匹配、特征点删除、计算变换矩阵、图像融合等。

为了方便运行测试，我们上传了sift-opencv249可执行文件sift_static.exe,其中运行参数在文件sift_static_exe_readme.txt中可以找到。

Test data set description：

Test images are stored in the test images folder.

1. PA-1 and PA-2 are  614×611 multispectral images. This image pair can be used to test the PSO-SIFT algorithm.
2. PB-1 and PB-2 are  617×593 multispectral images. This image pair can be used to test the PSO-SIFT algorithm.
3. Perspective_graf_1.ppm and perspective_graf_2.ppm are natural images with different shooting angles. This image pair can be used to test the SIFT algorithm. It should be pointed out that, perspective transformation model is needed for image pairs with different shooting angles.
4. Perspective_school_1.jpg and perspective_school_2.jpg are natural images with different shooting angles. This image pair can be used to test the SIFT algorithm.
5. SAR-SIFT_1.JPG and SAR-SIFT_2.JPG are SAR images. This image pair can be used to test the SAR-SIFT algorithm.


We regret to say that part of the test of remote sensing image do not belong to the public resources.




