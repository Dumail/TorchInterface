# TorchInterface
纯Java实现Pytorch模型的前向传播过程
## 应用背景
Pytorch是一种常用的神经网络框架，并且支持在Android设备使用训练好的模型。但在某些应用中，对框架的空间大小很严格，因此本项目以更轻量化的方式实现了部分Pytorch模型参数的读取，以及模型的前向传播过程。
## 实现模块
* Tensor：张量，Pytorch基础数据类型，对高纬度的支持尚未完善
* Linear：全连接层
* Conv2D：二维卷积层，支持输入三维数据（考虑到前向传播中常用于单条数据的预测，因此没加入Batch维度）
* Relu：relu激活函数层
* BatchNorm2D：二维归一化层
* MaxPool：最大池化层
* AveragePool：平均池化层
## 使用说明
相关接口与pytorch保持一致，网络模型可采用Layer.java中python函数保存参数到文件，然后新建一个继承自Network的类并搭建好相同的模型结构即可。可参考test中的相关测试用例。
