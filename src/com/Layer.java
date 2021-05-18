package com;

import java.util.Arrays;

/*
神经网络层
 */
public abstract class Layer {
    private float[] parameters; //网络参数
    private int inputSize; //输入节点数
    private int outputSize; //输出节点数

    public Layer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public void setParameters(float[] parameters) {
        this.parameters = parameters;
    }

    /*
    网络层正向传播
     */
    public Tensor forward(Tensor input) {
        return input;
    }
}
