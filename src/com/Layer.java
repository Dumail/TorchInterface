package com;

/**
 * 基础神经网络层，具有网络参数
 */
public abstract class Layer extends Module {
    protected Tensor parameters; //神经网络参数

    /**
     * 神经网络层构造函数
     * @param inputSize 输入节点数
     * @param outputSize 输出节点数
     */
    public Layer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    public void setParameters(Tensor parameters) {
        this.parameters = parameters;
    }
}
