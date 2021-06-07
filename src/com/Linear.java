package com;

import java.util.Arrays;

/*
全连接层
 */
public class Linear extends Layer {
    public Linear(int inputSize, int outputSize) {
        super(inputSize, outputSize);
        //矩阵形式，输入节点数x输出节点数，+1是由于将偏置也写入了矩阵
        parameters = new Tensor(outputSize, inputSize + 1);
        parameters.randomData(); //随机初始化权重
    }

    @Override
    public Tensor forward(Tensor input) {
        //输入数据的形状应该是（数据条数x特征维度）
        if (input.shape[input.dims() - 1] != inputSize) {
            System.out.println("Error" + Util.getPos() + " Input tensor size " + input.shape[input.dims() - 1] + " mismatch layer input size " + inputSize);
        }

        //将输入数据扩展以便于与权重偏置矩阵相乘
        int[] inputShape = Arrays.copyOf(input.shape, input.shape.length);
        inputShape[inputShape.length - 1] += 1;
        if (!input.expand(inputShape, 1))
            return null;

        return input.multi(parameters.T());
    }

    @Override
    public String toString() {
        return super.toString() + "-->Linear: \n input size: " + inputSize + ", output size: " + outputSize + "\n parameters:" + this.parameters;
    }

    @Override
    protected boolean loadParameters(ParametersTuple tuple) {
        int[] weight_shape = tuple.getWeight_shape();
        int[] bias_shape = tuple.getBias_shape();
        float[] weight = tuple.getWeight();
        float[] bias= tuple.getBias();

        if (weight_shape[1] != inputSize || outputSize != weight_shape[0]) {
            System.out.println("Error" + Util.getPos() + " weight " + Arrays.toString(weight_shape) + " mismatch input size " + this.inputSize + " or output size " + this.outputSize);
            return false;
        }

        if (weight_shape[0] != bias_shape[0]) {
            System.out.println("Error" + Util.getPos() + " weight " + Arrays.toString(weight_shape) + " mismatch bias " + Arrays.toString(bias_shape));
            return false;
        }

        float[] parameters = new float[weight_shape[0] * (weight_shape[1] + 1)];
        for (int i = 0; i < weight_shape[0]; i++) {
            //parameters比weight多一“列”
            System.arraycopy(weight, i * weight_shape[1], parameters, i * (weight_shape[1] + 1), weight_shape[1]);
            parameters[i * (weight_shape[1] + 1) + weight_shape[1]] = bias[i];
        }
//        System.arraycopy(weight, 0, parameters, 0, weight_shape[0] * weight_shape[1]);
//        System.arraycopy(bias, 0, parameters, weight_shape[0] * weight_shape[1], bias_shape[0]);
        setParameters(parameters);
        return true;
    }
}
