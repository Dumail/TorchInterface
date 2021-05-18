package com;

import java.util.Arrays;

/*
神经网络基本数据类型
 */
public class Tensor {
    public int[] shape; //张量形状
    private float[] data; //张量数据，采取一维数组存储

    public Tensor(float[] data) {
        this.data = new float[data.length];
        System.arraycopy(data, 0, this.data, 0, data.length);
        this.shape = new int[]{data.length};
    }

    public Tensor(float[][] data) {
        this.data = new float[data.length * data[0].length];
        for (int i = 0; i < data.length; i++)
            System.arraycopy(data[i], 0, this.data, i * data[0].length, data[0].length);
        this.shape = new int[]{data.length, data[0].length};
    }

    public boolean reshape(int[] shape) {
        this.shape = shape;
        return true;
    }
}
