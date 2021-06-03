package com;


/**
 * @program: TorchInterface
 * @description: 批标准化，默认批次为 1
 * @author: R.
 * @create: 2021-06-03-10-55
 **/


public class BatchNormal extends Layer{

    /**
     * 神经网络层构造函数
     *
     * @param inputSize  输入节点数
     * @param outputSize 输出节点数
     */
    public BatchNormal(int inputSize, int outputSize,int... args) {
        super(inputSize, outputSize);
    }




    @Override
    public Tensor forward(Tensor input) {
        return null;
    }
}
