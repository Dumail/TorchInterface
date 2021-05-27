package com;

import java.io.*;

/**
 * 基础神经网络层，具有网络参数
 */
public abstract class Layer extends Module {
    protected Tensor parameters; //神经网络参数

    /**
     * 神经网络层构造函数
     *
     * @param inputSize  输入节点数
     * @param outputSize 输出节点数
     */
    public Layer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    public void setParameters(Tensor parameters) {
        this.parameters = parameters;
    }

    public void setParameters(float[] parameters) {
        this.parameters.setData(parameters);
    }

    /**
     * 从字符串载入网络层参数
     *
     * @param parameters 字符串表示的参数
     * @return 是否读取成功
     */
    public boolean loadParameters(String parameters) {
        System.out.println("These parameters want to set, but this layer has no parameters:");
        System.out.println(parameters);
        System.out.println();
        return true;
    }

    /**
     * 从文件载入网络网络层参数
     *
     * @param filePath 参数文件
     * @return 是否读取成功
     */
    public boolean readParameters(String filePath) {
        String encoding = "UTF-8"; //已知文件编码
        File file = new File(filePath);
        long fileLength = file.length();
        //先全部读入字节流再一次性进行编码转换
        byte[] fileContent = new byte[(int) fileLength];
        try {
            FileInputStream in = new FileInputStream(file);
            int length = in.read(fileContent);
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            return this.loadParameters(new String(fileContent, encoding));
        } catch (UnsupportedEncodingException e) {
            System.err.println("The OS does not support " + encoding);
            e.printStackTrace();
            return false;
        }
    }
}
