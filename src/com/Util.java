package com;

/**
 * 工具类
 */
public class Util {
    /**
     * 获取一个形状的总大小
     * @param shape 形状
     * @return 总大小
     */
    public static int prod(int[] shape) {
        int temp = 1;
        for (int j : shape) temp *= j;
        return temp;
    }

    public static float[] relu(float[] input) {
        float[] output  = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] > 0 ? input[i] : 0 ;
        }
        return output;
    }
}
