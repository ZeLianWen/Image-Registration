该sift算法在opencv中的源代码基础上进行了个性化的定制，更加适合于图像配准任务。开发环境为VS2013+opencv2.4.9。该exe文件使用静态编译生成的,无需相应dll文件。

使用介绍：
需要通过命令行输入四个参数：exe文件，参考图像，待配准图像，变换模型。目前该算法支持相似变换“similarity”,仿射变换“affine”,透视变换“perspevtive”。如果第四个参数为空，默认是透视变换“perspevtive”。相比于opencv里面仅仅支持透视变换，该算法大大提高了灵活性。当前目录下面有两幅测试图像，这两幅测试图像之间存在透视变换关系，可以用来测试算法。如果模型之间存在相似变换关系，则三个模型都可以使用；如果图像之间存在仿射变换关系，则可以使用“透视变换”或者“仿射变换”模型；如果图像之间存在透视变换关系，则模型参数只能选择“透视变换”。

用法实例如下：
1.sift_2.exe image_1.jpg image_2.jpg similarity
2.sift_2.exe image_1.jpg image_2.jpg affine
3.sift_2.exe image_1.jpg image_2.jpg perspevtive
4.sift_2.exe image_1.jpg image_2.jpg 
第四种情况下，使用透视变换模型。

结果：
最终生成的配准结果保存在当前目录下的image_save文件夹。“配准后的参考图像”和“配准后的待配准图像”大小一致，相同坐标下对应同一个位置.

作者：西安电子科技大学，使用过程中遇到问题，请邮件联系：zelianwen@foxmail.com

