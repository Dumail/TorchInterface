package com;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/*
网络模型，包含多层网络
 */
public abstract class Network extends Module {

    public Module[] modules;

    /**
     * 构造函数，需要用户重写，搭建网络结构（即将需要的网络层放入this.modules）
     *
     * @param inputSize  输入数据维度
     * @param outputSize 输出数据维度
     */
    public Network(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    /**
     * 需要用户实现，也就是通过调用this.modules中每个module的forward方法来实现正向传播过程
     *
     * @param input 输入张量
     * @return 结果张量
     */
    public Tensor forward(Tensor input) {
        Tensor tempTensor = input;
        //逐层进行前向传播
        for (int i = 0; i < modules.length - 1; i++) {
            //对于其他层和全链接层的连接处，自动进行形状转换
            if (modules[i] instanceof Linear && (tempTensor.dims() != 2 || tempTensor.shape[1] != modules[i].inputSize))
                if (input.reshape(new int[]{-1, modules[i].inputSize})) {
                    System.out.println("Error: input tensor shape " + Arrays.toString(tempTensor.shape) + " of linear mismatch input size " + modules[i].inputSize + " of layer " + i);
                    return null;
                }
            tempTensor = modules[i].forward(tempTensor);
        }
        return tempTensor;
    }

    /**
     * 从文件载入网络网络层参数
     *
     * @param filePath 参数文件
     * @return 是否读取成功
     */
    public boolean readParameters(String filePath) {
        //从文件读取参数
        ArrayList<String> arrayList = new ArrayList<>();
        try {
            FileReader fr = new FileReader(filePath);
            BufferedReader bf = new BufferedReader(fr);
            String str;
            // 按行读取字符串
            while ((str = bf.readLine()) != null) {
                arrayList.add(str);
            }
            bf.close();
            fr.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        int numOfLayers = 0;//Layer的index
        for (int i = 0; i < this.modules.length; i++) {
            //对于无参数的层直接跳过
            if (!(modules[i] instanceof Layer))
                continue;

            //获取每个Layer的参数字符串
            StringBuilder buffer = new StringBuilder();
            for (int j = 0; j < 5; j++)
                if (numOfLayers * 6 + j < arrayList.size())
                    buffer.append(arrayList.get(numOfLayers * 6 + j)).append("\n");

            //设置每个Layer的参数
            ParametersTuple tuple;
            if ((tuple = ((Layer) modules[i]).paramStrProcess(buffer.toString())) == null) {
                System.out.println("Error: Something wrong when covert str to params at layer " + i);
                return false;
            }
            ((Layer) modules[i]).loadParameters(tuple);
            numOfLayers++;
        }
        return true;
    }

    @Override
    public String toString() {
        return "Network{" +
                "\n modules=" + Arrays.toString(modules) +
                '}';
    }
}