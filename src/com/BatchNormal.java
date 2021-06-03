package com;

import java.util.Arrays;

/**
 * @program: TorchInterface
 * @description: 批标准化，默认批次为 1
 * @author: R.
 * @create: 2021-06-03-10-55
 **/


public class BatchNormal extends Layer {
    protected float[] bias;
    protected int channel;

    /**
     * 神经网络层构造函数
     */
    public BatchNormal(int... args) {
        super(0, 0);
    }

    @Override
    public Tensor forward(Tensor input) {
        return null;
    }

    @Override
    public String toString() {
        return super.toString() + "-->BatchNormal{" +
                "bias=" + Arrays.toString(bias) +
                ", channel= " + channel + ", weight= " + Arrays.toString(parameters.getData()) +
                '}';
    }
}
