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
     * @param tuple 参数元组
     */
    protected boolean loadParameters(ParametersTuple tuple) {
        System.out.println("Error" + Util.getPos() + "These parameters want to set, but this layer has no parameter!");
        System.out.println(parameters);
        System.out.println();
        return false;
    }

    /**
     * 从二进制文件载入网络层参数
     * 参数存储文件通过以下python函数存储：
     * def save_visual(net):
     * par = net.parameters()
     * list_par = []
     * for p in par:
     * temp_list=[]
     * shape = list(p.shape)
     * len_shape = len(p.shape)
     * data = p.data.flatten().numpy().tolist()
     * len_data = len(data)
     * temp_list.append(len_shape)
     * temp_list.extend(shape)
     * temp_list.append(len_data)
     * temp_list.extend(data)
     * list_par.append(temp_list)
     * with open("pars.pt", "wb") as f:
     * for par in list_par:
     * f.write(struct.pack('>i',par[0]))
     * #             print(par[0])
     * for i in range(par[0]):
     * f.write(struct.pack('>i',par[i+1]))
     * #                 print(par[i+1])
     * f.write(struct.pack('>i',par[par[0]+1]))
     * #             print(par[par[0]+1])
     * for i in range(par[par[0]+1]):
     * f.write(struct.pack('>f',par[par[0]+2+i]))
     * #                 print(par[par[0]+2+i])
     *
     * @param filePath 参数文件路径
     * @return 是否读取成功
     */
    public boolean readParameters(String filePath) {
        File file = new File(filePath);
        ParametersTuple tuple = new ParametersTuple();
        if (!file.exists()) {
            System.out.println("Error" + Util.getPos() + " Open file failed.");
            return false;
        }
        try {
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
            // 权重读取
            int weight_shape_size = in.readInt();
            int[] weight_shape = new int[weight_shape_size];
            for (int j = 0; j < weight_shape_size; j++)
                weight_shape[j] = in.readInt();
            int weight_data_size = in.readInt();
            float[] weight_data = new float[weight_data_size];
            for (int j = 0; j < weight_data_size; j++)
                weight_data[j] = in.readFloat();

            // 偏置读取
            int bias_shape_size = in.readInt();
            int[] bias_shape = new int[bias_shape_size];
            for (int j = 0; j < bias_shape_size; j++)
                bias_shape[j] = in.readInt();
            int bias_data_size = in.readInt();
            float[] bias_data = new float[bias_data_size];
            for (int j = 0; j < bias_data_size; j++)
                bias_data[j] = in.readFloat();
            tuple.setParameters(weight_shape, bias_shape, weight_data, bias_data);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return loadParameters(tuple);
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