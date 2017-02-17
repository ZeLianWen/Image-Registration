# Image-Registration
Image registration algorithm. Includes SIFT, SAR-SIFT,PSO-SIFT.

Author information：
Key Laboratory of Intelligent Perception and Image Understanding of Ministry of Education, International Research Center for Intelligent Perception and Computation, Joint International Research　Laboratory of Intelligent Perception and Computation, Xidian University, Xi’an, Shaanxi Province, 710071, China(email:zelianwen@foxmail.com).

Algorithm description：

1. SIFT(Scale-invariant feature transform). Reference article: David G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints"[2004].
2. SAR-SIFT. Reference article: Flora Dellinger, Julie Delon, Yann Gousseau, Julien Michel, and Florence Tupin, "SAR-SIFT: A SIFT-Like Algorithm for SAR Images"[2015].
3. PSO-SIFT. Reference article: "Remote Sensing Image Registration with Modified SIFT and Enhanced Feature Matching"[2016].If you do use it, please cite:

Ma W, Wen Z, Wu Y, et al. Remote Sensing Image Registration With Modified SIFT and Enhanced Feature Matching[J]. IEEE Geoscience and Remote Sensing Letters, 2017, 14(1): 3-7.


Test data set description：

Test images are stored in the test images folder.

1. PA-1 and PA-2 are  614×611 multispectral images. This image pair can be used to test the PSO-SIFT algorithm.
2. PB-1 and PB-2 are  617×593 multispectral images. This image pair can be used to test the PSO-SIFT algorithm.
3. Perspective_graf_1.ppm and perspective_graf_2.ppm are natural images with different shooting angles. This image pair can be used to test the SIFT algorithm. It should be pointed out that, perspective transformation model is needed for image pairs with different shooting angles.
4. Perspective_school_1.jpg and perspective_school_2.jpg are natural images with different shooting angles. This image pair can be used to test the SIFT algorithm.
5. SAR-SIFT_1.JPG and SAR-SIFT_2.JPG are SAR images. This image pair can be used to test the SAR-SIFT algorithm.


We regret to say that part of the test of remote sensing image do not belong to the public resources.

