package com;

import java.util.Arrays;

public class Conv2D extends ConvNd {

    /**
     * 二维卷积层构造函数
     *
     * @param inputSize  输入节点数
     * @param outputSize 输出节点数
     * @param args       依次为卷积核大小，填充大小，步长
     */
    public Conv2D(int inputSize, int outputSize, int[]... args) {
        super(inputSize, outputSize, args);
        if (args[0].length != 2)
            System.out.println("Error" + Util.getPos() + " args dim must be 2.");

        //卷积核参数矩阵, 第一维是卷积核个数，第二维是通道数与输入一致，第三第四维为自定义卷积核大小
        parameters = new Tensor(outputSize, inputSize, this.kernelSize[0], this.kernelSize[1]);
        parameters.randomData(); //随机初始化权重
    }

    @Override
    public Tensor forward(Tensor input) {
        //二维卷积
        if (input.dims() != 2 && input.dims() != 3) {
            System.out.println("Error" + Util.getPos() + " Current version does not support high dim to conv.");
            return null;
        }

        if (input.dims() == 2)
            input.unSqueeze(0);
        int channel = input.shape[0];
        if (this.inputSize != channel) {
            System.out.println("Error" + Util.getPos() + " input tensor channel " + channel + " mismatch this layer's input size " + inputSize);
            return null;
        }

        int height = (input.shape[1] + 2 * this.padding[0] - kernelSize[0]) / this.step[0] + 1;
        int weight = (input.shape[2] + 2 * this.padding[1] - kernelSize[1]) / this.step[1] + 1;
        int inputHeight = input.shape[1] + this.padding[0] * 2;
        int inputWeight = input.shape[2] + this.padding[1] * 2;

        float[][][] outData = new float[this.outputSize][height][weight];
        float[][][] inputDataRaw = input.getData3D();//原始输入数组
        //padding后的输入数组
        float[][][] inputData = new float[channel][inputHeight][inputWeight];
        float[][][][] kernelData = this.parameters.getData4D();

        //输入数据填充 TODO 其他填充类型
        for (int i = 0; i < channel; i++)
            for (int j = this.padding[0]; j < inputHeight; j++)
                System.arraycopy(inputDataRaw[i][j], 0, inputData[0][j], this.padding[1], inputDataRaw[i].length);

        //卷积操作 TODO 复杂度更低的算法
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < weight; w++) {
                    float sum = 0;
                    for (int c = 0; c < channel; c++) {
                        for (int kh = 0; kh < kernelSize[0]; kh++) {
                            for (int kw = 0; kw < kernelSize[1]; kw++) {
                                sum += inputData[c][h * this.step[0] + kh][w * this.step[1] + kw] * kernelData[o][c][kh][kw];
                            }
                        }
                    }
                    outData[o][h][w] = sum;
                }
            }
        }

        Tensor outTensor = new Tensor(outData);

        //加上偏置
        for (float bia : this.bias) outTensor.add(bia);

        //去掉多余维度，以方便其他层（例如全连接）计算
        outTensor.squeeze();
        return outTensor;
    }

    @Override
    protected boolean loadParameters(ParametersTuple tuple) {
        int[] weight_shape = tuple.getWeight_shape();
        int[] bias_shape = tuple.getBias_shape();
        float[] weight = tuple.getWeight();
        float[] bias = tuple.getBias();

        //卷积核参数需要是四维的
        if (weight_shape.length != 4) {
            System.out.println("Error" + Util.getPos() + " Weight of conv2D must be 4 dimension, but this weight dimension is " + weight_shape.length);
            return false;
        }
        if (weight_shape[0] != outputSize || weight_shape[1] != inputSize || weight_shape[2] != kernelSize[0] || weight_shape[3] != kernelSize[1]) {
            System.out.println("Error" + Util.getPos() + " Parameters of this file " + Arrays.toString(weight_shape) + " mismatch this conv2D layer.");
            return false;
        }
        if (bias_shape[0] != outputSize) {
            System.out.println("Error" + Util.getPos() + " Bias number of this file " + bias_shape[0] + " mismatch this layer's output size " + outputSize);
            return false;
        }

        //复制参数
        int length = Util.prod(weight_shape);
        float[] weightData = new float[length];
        System.arraycopy(weight, 0, weightData, 0, length);
        System.arraycopy(bias, 0, this.bias, 0, bias_shape[0]);

        this.bias = bias;
        setParameters(weightData);
        return true;
    }
}
