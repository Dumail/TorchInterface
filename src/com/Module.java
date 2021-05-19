package com;

import java.util.Arrays;

/*
神经网络层
 */
public abstract class Module {
    protected int inputSize; //输入节点数
    protected int outputSize; //输出节点数

    public Module(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public abstract Tensor forward(Tensor input);
}

