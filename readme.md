# 手写字体分类和识别
本项目手写字体来自于搜集同学书写的500个字

使用datasets.ImageFolder来进行对多文件夹的图片进行加载和类别标注
，实现了分类和识别准确率高达98.6%

e.g. 

train_dataset = datasets.ImageFolder('./data/train', transform=data_transform)

程序运行过程见ius.txt