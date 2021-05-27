package com;

import java.util.Arrays;
import java.util.Random;

/*
全连接层
 */
public class Linear extends Layer {
    public Linear(int inputSize, int outputSize) {
        super(inputSize, outputSize);
        //矩阵形式，输入节点数x输出节点数，+1是由于将偏置也写入了矩阵
        parameters = new Tensor(inputSize + 1, outputSize);
        parameters.randomData(); //随机初始化权重
    }

    @Override
    public Tensor forward(Tensor input) {
        //输入数据的形状应该是（特征维度x输入节点数）
        if (input.shape[input.shape.length - 1] != inputSize) {
            System.out.println("Error: Input tensor size " + input.shape[input.shape.length - 1] + " mismatch layer input size " + inputSize);
        }

        //将输入数据扩展以便于与权重偏置矩阵相乘
        int[] inputShape = Arrays.copyOf(input.shape, input.shape.length);
        inputShape[inputShape.length - 1] += 1;
        if (!input.expand(inputShape, 1))
            return null;

        return input.multi(parameters);
    }

    @Override
    public String toString() {
        return super.toString() + "Linear: \n input size: " + inputSize + ", output size: " + outputSize + "\n parameters:" + this.parameters;
    }

    @Override
    public boolean loadParameters(String parameters_str) {
        String[] listParameters = parameters_str.split("\n");
        int[] weight_shape, bias_shape;
        float[] weight, bias;
        try {
            //得到权重形状数组
            weight_shape = Arrays.stream(listParameters[0].substring(1, listParameters[0].length() - 1).replace(" ", "").split(",")).mapToInt(Integer::parseInt).toArray();
            //得到权重数组
            String[] weight_str = listParameters[1].substring(1, listParameters[1].length() - 1).trim().replace("  ", " ").split(" ");
            weight = new float[weight_str.length];
            for (int i = 0; i < weight_str.length; i++) {
                weight[i] = Float.parseFloat(weight_str[i]);
            }

            //得到偏置形状数组
            bias_shape = Arrays.stream(listParameters[3].substring(1, listParameters[3].length() - 1).replace(" ", "").split(",")).mapToInt(Integer::parseInt).toArray();
            //得到偏置数组
            String[] bias_str = listParameters[4].substring(1, listParameters[4].length() - 1).trim().replace("  ", " ").split(" ");
            bias = new float[bias_str.length];
            for (int i = 0; i < bias_str.length; i++) {
                bias[i] = Float.parseFloat(bias_str[i]);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }

        if (weight_shape[0] != inputSize || outputSize != weight_shape[1]) {
            System.out.println("Error: weight " + Arrays.toString(weight_shape) + " mismatch input size " + this.inputSize + " or output size " + this.outputSize);
            return false;
        }

        if (weight_shape[0] != bias_shape[0]) {
            System.out.println("Error: weight " + Arrays.toString(weight_shape) + " mismatch bias " + Arrays.toString(bias_shape));
            return false;
        }

        float[] parameters = new float[weight_shape[0] * (weight_shape[1] + 1)];
        for (int i = 0; i < weight_shape[0]; i++) {
            System.arraycopy(weight, i * weight_shape[1], parameters, i * (weight_shape[1] + 1), weight_shape[1]);
            parameters[i * (weight_shape[1] + 1) + weight_shape[1]] = bias[i];
        }
        setParameters(parameters);
        return true;
    }
}
