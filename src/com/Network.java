package com;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
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
        for (int i = 0; i < modules.length; i++) {
            if (tempTensor == null) {
                System.out.println("Error" + Util.getPos() + " Some thing wrong when forward at layer " + (i - 1));
                return null;
            }
            //对于其他层和全链接层的连接处，自动进行形状转换
            if (modules[i] instanceof Linear && (tempTensor.dims() != 2 || tempTensor.shape[1] != modules[i].inputSize))
                if ((tempTensor = tempTensor.reshape(new int[]{-1, modules[i].inputSize})) == null) {
                    System.out.println("Error" + Util.getPos() + " input tensor shape of linear mismatch input size " + modules[i].inputSize + " of layer " + i);
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
        File file = new File(filePath);
        if (!file.exists()) {
            System.out.println("Error" + Util.getPos() + " Open file failed.");
            return false;
        }
        try {
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
            for (Module module : this.modules) {
                //对于无参数的层直接跳过
                if (!(module instanceof Layer))
                    continue;

                ParametersTuple tuple = new ParametersTuple();
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
                ((Layer) module).loadParameters(tuple);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
//        //从文件读取参数
//        ArrayList<String> arrayList = new ArrayList<>();
//        try {
//            FileReader fr = new FileReader(filePath);
//            BufferedReader bf = new BufferedReader(fr);
//            String str;
//            // 按行读取字符串
//            while ((str = bf.readLine()) != null) {
//                arrayList.add(str);
//            }
//            bf.close();
//            fr.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        int numOfLayers = 0;//Layer的index
//        for (int i = 0; i < this.modules.length; i++) {
//            //对于无参数的层直接跳过
//            if (!(modules[i] instanceof Layer))
//                continue;
//
//            //获取每个Layer的参数字符串
//            StringBuilder buffer = new StringBuilder();
//            for (int j = 0; j < 5; j++)
//                if (numOfLayers * 6 + j < arrayList.size())
//                    buffer.append(arrayList.get(numOfLayers * 6 + j)).append("\n");

        //设置每个Layer的参数
//            ParametersTuple tuple;
//            if ((tuple = ((Layer) modules[i]).paramStrProcess(buffer.toString())) == null) {
//                System.out.println("Error" + Util.getPos() + " Something wrong when covert str to params at layer " + i);
//                return false;
//            }
//            ((Layer) modules[i]).loadParameters(tuple);
//            numOfLayers++;
//        }
        return true;
    }

    @Override
    public String toString() {
        return "Network{" +
                "\n modules=" + Arrays.toString(modules) +
                '}';
    }
}