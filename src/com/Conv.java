package com;


/**
 * 卷积层
 */
public class Conv extends Layer {
    int kernelSize;//卷积核大小
    int padding = 0; //填充
    int step = 1;//步长

    /**
     * 卷积层构造函数
     *
     * @param inputSize  输入节点数
     * @param outputSize 输出节点数
     */
    public Conv(int inputSize, int outputSize, int... args) {
        super(inputSize, outputSize);
        //卷积核参数矩阵
        parameters = new Tensor(inputSize + 1, outputSize);
        parameters.randomData(); //随机初始化权重

        this.kernelSize = args[0];
        if (args.length > 1) this.padding = args[1];
        if (args.length > 2) this.step = args[2];
    }


    @Override
    public Tensor forward(Tensor input) {
        return null;
    }
}
