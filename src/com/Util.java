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
}
