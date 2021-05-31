package com;


import java.util.Arrays;

/**
 * 卷积层
 */
public abstract class ConvNd extends Layer {
    protected int[] kernelSize;//卷积核大小
    protected int[] padding; //填充
    protected int[] step;//步长
    protected String paddingType = "zero"; //填充类型，补零
    protected float[] bias;

    /**
     * 卷积层构造函数
     *
     * @param inputSize  输入节点数
     * @param outputSize 输出节点数
     * @param args       包含卷积核大小和可选参数（填充大小，步长），都使用数组进行传递，数组的第几个元素就表示第几维的参数
     */
    public ConvNd(int inputSize, int outputSize, int[]... args) {
        super(inputSize, outputSize);

        int dim = args[0].length; //卷积维度
        this.kernelSize = Arrays.copyOf(args[0], dim);
        if (args.length > 1)
            this.padding = Arrays.copyOf(args[1], dim);
        else
            this.padding = new int[dim];
        if (args.length > 2)
            this.step = Arrays.copyOf(args[2], dim);
        else {
            this.step = new int[dim];
            Arrays.fill(this.step, 1);
        }
        this.bias = new float[dim];
        //TODO 随机初始化
    }


    @Override
    public String toString() {
        return super.toString() + "Conv: \n input size: " + inputSize + ", output size: " + outputSize + "\n  kernel size: " + Arrays.toString(this.kernelSize) + "\n  padding: " + Arrays.toString(this.padding) + "\n  step: " + Arrays.toString(this.step) + "\n  weight:" + this.parameters + "\n  bias: " + Arrays.toString(this.bias);
    }
}