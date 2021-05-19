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
        parameters = new Tensor(new int[]{inputSize + 1, outputSize});
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
}
