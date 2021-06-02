package com;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
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
    protected boolean loadParameters(ParametersTuple tuple) {
        System.out.println("These parameters want to set, but this layer has no parameters:");
        System.out.println(parameters);
        System.out.println();
        return false;
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
        if (!file.exists()) {
            System.out.println("Error" + Util.getPos() + " Open file failed.");
            return false;
        }
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
            paramStr = new String(fileContent, encoding);
        } catch (UnsupportedEncodingException e) {
            System.err.println("The OS does not support " + encoding);
            e.printStackTrace();
            return false;
        }

        ParametersTuple tuple;
        if ((tuple = paramStrProcess(paramStr)) == null) {
            System.out.println("Error" + Util.getPos() + " Something wrong when covert str to params.");
            return false;
        }
        return loadParameters(tuple);
    }

    /**
     * 处理参数字符串得到参数元祖。
     * 对于单个网络层，可以将参数文件中所有字符串用该方法处理
     * 对于多层网络，需要将各层的参数字符串分别用该方法处理
     * @param paramStr 字符串形式的参数
     * @return 元组形式的参数
     */
    protected ParametersTuple paramStrProcess(String paramStr) {
        ParametersTuple tuple = new ParametersTuple();
        String[] listParameters = paramStr.split("\n");
        int[] weight_shape, bias_shape;
        float[] weight, bias;
        try {
            //得到权重形状数组
            weight_shape = Arrays.stream(listParameters[0].substring(1, listParameters[0].length() - 1).replace(" ", "").split(",")).mapToInt(Integer::parseInt).toArray();
            //得到权重数组
            String[] weight_str = listParameters[1].trim().split(",");
            weight = new float[weight_str.length];
            for (int i = 0; i < weight_str.length; i++) {
                weight[i] = Float.parseFloat(weight_str[i]);
            }

            //得到偏置形状数组
            bias_shape = Arrays.stream(listParameters[3].substring(1, listParameters[3].length() - 1).replace(" ", "").split(",")).mapToInt(Integer::parseInt).toArray();
            //得到偏置数组
            String[] bias_str = listParameters[4].trim().split(",");
            bias = new float[bias_str.length];
            for (int i = 0; i < bias_str.length; i++) {
                bias[i] = Float.parseFloat(bias_str[i]);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        tuple.setParameters(weight_shape, bias_shape, weight, bias);
        return tuple;
    }
}

/**
 * 参数元祖，用于方便参数返回
 */
class ParametersTuple {
    float[] weight, bias; //实际参数
    int[] weight_shape, bias_shape; //权重和偏置的形状

    public void setParameters(int[] weight_shape, int[] bias_shape, float[] weight, float[] bias) {
        this.weight_shape = weight_shape;
        this.bias_shape = bias_shape;
        this.weight = weight;
        this.bias = bias;
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