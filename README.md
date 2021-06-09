# TorchInterface
纯Java实现Pytorch模型的前向传播过程
## 应用背景
Pytorch是一种常用的神经网络框架，并且支持在Android设备上使用训练好的模型。但在某些应用中，对神经网络框架的空间占用有很严格的限制，因此本项目以更轻量化的方式实现了部分Pytorch模型参数的读取，以及模型的前向传播过程。
## 实现模块
* Tensor：张量，Pytorch基础数据类型，对高纬度的支持尚未完善
* Linear：全连接层
* Conv2D：二维卷积层，支持输入三维数据（考虑到前向传播常用于单条数据的预测，因此没加入Batch维度）
* Relu：relu激活函数层
* BatchNorm2D：二维归一化层
* MaxPool：最大池化层
* AveragePool：平均池化层
## 使用说明
相关接口与pytorch保持一致，网络模型可采用下列函数保存参数到文件：
```python
def save_net(net):
	par = net.parameters()
 	list_par = []
    for p in par:
        shape = list(p.shape)
        len_shape = len(p.shape)
        data = p.data.flatten().numpy().tolist()
        len_data = len(data)
        list_par.append([len_shape]+shape+[len_data]+data)
    with open("pars.pt", "wb") as f:
        for par in list_par:
            f.write(struct.pack('>i',par[0]))
            for i in range(par[0]):
                f.write(struct.pack('>i',par[i+1]))
                f.write(struct.pack('>i',par[par[0]+1]))
                for i in range(par[par[0]+1]):
                    f.write(struct.pack('>f',par[par[0]+2+i]))
```
然后新建一个继承自Network的类即可:
```java
class MyNet extends Network {
        public MyNet(int inputSize, int outputSize) {
            super(inputSize, outputSize);
            this.modules = new Module[]{
                    new Conv2D(inputSize, 128, new int[]{3, 3}),
                    new BatchNorm2d(128),
                    new MaxPool(new int[]{3, 3}),
                    new Relu(),
                    new Linear(1024, outputSize)
            };
        }
}
```
注意，需要在自建类的构造函数中建立与Python端相同的网络结构，且网络结构需要存储于类的modules中（否则需要重写读取网络参数的方法）。为了方便，Network的默认forward方法中实现了卷积层到全连接层的自动Reshape，因此此类简单的网络可以直接用于预测。对于其他情况，仍需重写其forward方法。
预测时，需要先调用MyNet对象的readParameters方法读取相应参数，然后将相应的数据包装成Tensor对象（使用一维或多维数组构造），最后调用该网络对象的forward方法进行预测。
