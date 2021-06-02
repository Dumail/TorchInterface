package com;

/**
 * 工具类
 */
public class Util {
    /**
     * 获取一个形状的总大小
     *
     * @param shape 形状
     * @return 总大小
     */
    public static int prod(int[] shape) {
        int temp = 1;
        for (int j : shape) temp *= j;
        return temp;
    }

    public static float[] relu(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
        return output;
    }

    /**
     * 生成全为1的张量
     *
     * @param shape 数据的形状
     * @return 全1张量
     */
    public static Tensor onesTensor(int[] shape) {
        Tensor temp = new Tensor(shape);
        temp.onesData();
        return temp;
    }

    /**
     * 生成指定范围内一维张量，步长为1
     *
     * @param start 开始数
     * @param end   结束数
     * @return 一维张量
     */
    public static Tensor rangeTensor(int start, int end) {
        float[] temp = new float[end - start];
        if (end <= start) {
            System.out.println("Warning" + Util.getPos() + ": range start must smaller than end.");
        }
        int index = 0;
        for (int i = start; i < end; i++)
            temp[index++] = i;
        return new Tensor(temp);
    }

    public static String getPos() {
        StackTraceElement temp = Thread.currentThread().getStackTrace()[2];
        return "(" + temp.getFileName() + "->" + temp.getClassName() + "->" + temp.getMethodName() + "->line" + temp.getLineNumber() + "): ";
    }

}
