package com;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/*
神经网络基本数据类型：张量
 */
public class Tensor {
    public int[] shape; //张量形状
    private float[] data; //张量数据，采取一维数组存储

    /**
     * 从数据形状构造张量
     *
     * @param shape 形状数组
     */
    public Tensor(int[] shape) {
        this.shape = new int[shape.length];
        System.arraycopy(shape, 0, this.shape, 0, shape.length);
        this.data = new float[Util.prod(shape)];
    }

    /**
     * 从数据形状构造张量
     *
     * @param shapeList 形状列表
     */
    public Tensor(ArrayList<Integer> shapeList) {
        this.shape = new int[shapeList.size()];
        for (int i = 0; i < shapeList.size(); i++)
            shape[i] = shapeList.get(i);
        this.data = new float[Util.prod(shape)];
    }

    /**
     * 从一维数据构造张量
     *
     * @param data 数组数据
     */
    public Tensor(float[] data) {
        this.data = new float[data.length];
        System.arraycopy(data, 0, this.data, 0, data.length);
        this.shape = new int[]{data.length};
    }

    /**
     * 数据化数据
     */
    public void randomData() {
        Random rand = new Random();
        for (int i = 0; i < data.length; i++)
            data[i] = rand.nextFloat();
    }

    /**
     * 从二维数据构造张量
     *
     * @param data 数组数据
     */
    public Tensor(float[][] data) {
        this.data = new float[data.length * data[0].length];
        // 逐层复制数据
        for (int i = 0; i < data.length; i++)
            System.arraycopy(data[i], 0, this.data, i * data[0].length, data[0].length);
        this.shape = new int[]{data.length, data[0].length};
    }

    /**
     * 改变张量形状
     *
     * @param shape 新的数据形状
     * @return 是否改变成功
     */
    public boolean reshape(int[] shape) {
        if (Util.prod(shape) != data.length) {
            System.out.println("Error: Data size " + data.length + " can not reshape to " + Arrays.toString(shape) + "!");
            return false;
        }
        System.arraycopy(shape, 0, this.shape, 0, shape.length);
        return true;
    }

    /**
     * 将张量扩展为指定形状，扩展的部分填充指定数据
     *
     * @param shape 扩展后形状
     * @param pad   填充数据
     * @return 是否扩展成功
     */
    public boolean expand(int[] shape, float pad) {
        if (shape.length != this.shape.length) {
            System.out.println("Error: Shape dimension " + this.shape.length + " mismatch " + shape.length);
            return false;
        }
        for (int i = 0; i < shape.length; i++) {
            if (this.shape[i] > shape[i]) {
                System.out.println("Error: Shape " + Arrays.toString(this.shape) + " can not be less than " + Arrays.toString(shape));
                return false;
            }
        }
        //TODO expand
        return true;
    }

    /**
     * 矩阵乘法，将张量的最后两维与另一个张量的最后两维相乘
     *
     * @param input 另一个张量
     * @return 乘法结果
     */
    public Tensor multi(Tensor input) {
        //相乘张量至少有两维
        if (shape.length < 2 || input.shape.length < 2) {
            System.out.println("Error: Tensor dimension is too low.");
        }
        //两个张量维度需要相同
        if (shape.length != input.shape.length) {
            System.out.println("Error: Dimension " + shape.length + " mismatch " + input.shape.length);
            return null;
        }

        //高于二维的形状需要相同
        for (int i = 0; i < shape.length - 2; i++) {
            if (shape[i] != input.shape[i]) {
                System.out.println("Error: Multiplied shape " + Arrays.toString(shape) + " by shape " + Arrays.toString(input.shape));
                return null;
            }
        }

        //相乘矩阵的行和列
        int rowOfMat1 = shape[shape.length - 2];
        int colOfMat1 = shape[shape.length - 1];
        int rowOfMat2 = input.shape[input.shape.length - 2];
        int colOfMat2 = input.shape[input.shape.length - 1];

        //相乘的两维行与列相同
        if (colOfMat1 != rowOfMat2) {
            System.out.println("Error: Multiplied shape " + Arrays.toString(shape) + " by shape " + Arrays.toString(input.shape));
            return null;
        }

        //实例化输出张量
        int[] outShape = Arrays.copyOf(shape, shape.length);
        outShape[outShape.length - 1] = colOfMat2;
        Tensor output = new Tensor(outShape);

        //TODO 矩阵相乘
        return null;
    }

}
