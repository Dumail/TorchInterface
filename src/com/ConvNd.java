package com;


import java.util.Arrays;

/**
 * 卷积层
 */
public abstract class ConvNd extends Layer {
    int[] kernelSize;//卷积核大小
    int[] padding; //填充
    int[] step;//步长
    String paddingType = "zero"; //填充类型，补零

    /**
     * 卷积层构造函数
     *
     * @param inputSize  输入节点数
     * @param outputSize 输出节点数
     * @param args       包含卷积核大小和可选参数（填充大小，步长），都使用数组进行传递，数组的第几个元素就表示第几维的参数
     */
    public ConvNd(int inputSize, int outputSize, int[]... args) {
        super(inputSize, outputSize);
        //卷积核参数矩阵
        parameters = new Tensor(inputSize + 1, outputSize);
        parameters.randomData(); //随机初始化权重

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
    }
}

class Conv2D extends ConvNd {

    /**
     * 二维卷积层构造函数
     */
    public Conv2D(int inputSize, int outputSize, int[]... args) {
        super(inputSize, outputSize, args);
        if (args[0].length != 2)
            System.out.println("Error: args dim must be 2.");
    }

    @Override
    public Tensor forward(Tensor input) {
        //二维张量的卷积
        if (input.dims() != 2 && input.dims() != 3) {
            System.out.println("Error: Current version does not support high dim to conv.");
            return null;
        }

        if (input.dims() == 2)
            input.reshape(new int[]{1, input.shape[0], input.shape[1]});
        int channel = input.shape[0];
        int height = (input.shape[1] + 2 * this.padding[0] - input.shape[1]) / this.step[0] + 1;
        int weight = (input.shape[2] + 2 * this.padding[1] - input.shape[2]) / this.step[1] + 1;
        int inputHeight = input.shape[1] + this.padding[0] * 2;
        int inputWeight = input.shape[2] + this.padding[1] * 2;

//        float[][] outData = new float[height][weight];
//        float[][] inputDataRaw = input.getData2D();//原始输入数组
//        //padding后的输入数组
//        float[][] inputData = new float[inputHeight][inputWeight];
//        float[][] kernelData = this.parameters.getData2D();
//
//        //TODO 其他填充类型
//        for (int i = this.padding[0]; i < inputHeight; i++)
//            System.arraycopy(inputDataRaw[i], 0, inputData[0], 0, inputDataRaw.length);
//
//        for (int h = 0; h < height; h += this.step) {
//            for (int w = 0; w < weight; w += this.step) {
//                float sum = 0;
//                for (int kh = 0; kh < inputHeight; kh++) {
//                    for (int kw = 0; kw < inputWeight; kw++) {
//                        sum += inputData[h + kh][w + kw] * kernelData[kh][kw];
//                    }
//                }
//                outData[h][w] = sum;
//            }
//        }
//        return new Tensor(outData);
        return null;
    }
}