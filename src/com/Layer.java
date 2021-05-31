package com;

import java.io.*;
import java.util.Arrays;

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
     * @param tuple 字符串表示的参数
     */
    public void loadParameters(ParametersTuple tuple) {
        System.out.println("These parameters want to set, but this layer has no parameters:");
        System.out.println(parameters);
        System.out.println();
    }

    /**
     * 从文件载入网络网络层参数
     *
     * @param filePath 参数文件
     * @return 是否读取成功
     */
    public boolean readParameters(String filePath) {
        ParametersTuple tuple = new ParametersTuple();

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

        //编码转换
        String paramStr;
        try {
//            this.loadParameters(new String(fileContent, encoding));
            paramStr=new String(fileContent,encoding);
        } catch (UnsupportedEncodingException e) {
            System.err.println("The OS does not support " + encoding);
            e.printStackTrace();
            return false;
        }

        String[] listParameters = paramStr.split("\n");
        int[] weight_shape, bias_shape;
        float[] weight, bias;
        try {
            //得到权重形状数组
            weight_shape = Arrays.stream(listParameters[0].substring(1, listParameters[0].length() - 1).replace(" ", "").split(",")).mapToInt(Integer::parseInt).toArray();
            //得到权重数组
            String[] weight_str = listParameters[1].substring(1, listParameters[1].length() - 1).trim().replace("  ", " ").split(" ");
            weight = new float[weight_str.length];
            for (int i = 0; i < weight_str.length; i++) {
                weight[i] = Float.parseFloat(weight_str[i]);
            }

            //得到偏置形状数组
            bias_shape = Arrays.stream(listParameters[3].substring(1, listParameters[3].length() - 1).replace(" ", "").split(",")).mapToInt(Integer::parseInt).toArray();
            //得到偏置数组
            String[] bias_str = listParameters[4].substring(1, listParameters[4].length() - 1).trim().replace("  ", " ").split(" ");
            bias = new float[bias_str.length];
            for (int i = 0; i < bias_str.length; i++) {
                bias[i] = Float.parseFloat(bias_str[i]);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }

        if (weight_shape[1] != inputSize || outputSize != weight_shape[0]) {
            System.out.println("Error: weight " + Arrays.toString(weight_shape) + " mismatch input size " + this.inputSize + " or output size " + this.outputSize);
            return false;
        }

        if (weight_shape[0] != bias_shape[0]) {
            System.out.println("Error: weight " + Arrays.toString(weight_shape) + " mismatch bias " + Arrays.toString(bias_shape));
            return false;
        }
        
        tuple.setParameters(weight_shape,bias_shape,weight,bias);
        loadParameters(tuple);
        return true;
    }
}

class ParametersTuple {
    float[] weight, bias;
    int[] weight_shape, bias_shape;

    public void setParameters(int[] weight_shape,int[] bias_shape,float[] weight,float[] bias)
    {
        this.weight_shape=weight_shape;
        this.bias_shape=bias_shape;
        this.weight=weight;
        this.bias=bias;
    }

    public float[] getWeight() {
        return weight;
    }

    public float[] getBias() {
        return bias;
    }

    public int[] getWeight_shape() {
        return weight_shape;
    }

    public int[] getBias_shape() {
        return bias_shape;
    }
}