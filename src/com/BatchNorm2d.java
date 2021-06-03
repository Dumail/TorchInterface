package com;


import java.util.Arrays;

/**
 * @program: TorchInterface
 * @description: 2d 标准化 ,  参考博客 ： https://blog.csdn.net/weixin_44278406/article/details/105554268
 * @author: R.
 * @create: 2021-06-03-11-00
 **/


public class BatchNorm2d  extends  BatchNormal{
    /**
     * 神经网络层构造函数
     *
     * @param inputSize  输入节点数
     * @param outputSize 输出节点数
     */
    float[] Weight;
    float[] Bias;
    int inputSize;
    int outputSize;
    int Channel;
    int[] shape = new int[]{0,0};
    /**
     *
     * @param inputSize  输入节点个数
     * @param outputSize  输出节点个数默认与 input 相同
     * @param args      第一个参数为 通道个数C，W 默认为1 ，B 默认为0
     */


    public BatchNorm2d(int inputSize,int outputSize,int... args) {
        super(inputSize,0);

        if (args.length != 1)
            System.out.println("Error: BatchNorm2d args must be 1."+args.length);
        Channel = args[0];
        Weight  = new float[Channel];
        for (int i = 0; i < Channel; i++) {
            Weight[i] = (float) 1.0;
        }
        Bias  = new float[Channel];
        for (int i = 0; i < Channel; i++) {
            Bias[i] = 0 ;
        }



//            if(args[1].length == 2)
//                shape =  Arrays.copyOf(args[1], args[1].length);

    }

    public void initWB(Tensor W,Tensor B){      // 初始化 偏移量W与附加量B ，若有该参数则调用该初始化方法，没有则不用。
        if (Weight.length != Channel || Bias.length != Channel || Bias.length != Weight.length )
            System.out.println("Error: BatchNorm2d Weight and Bias must be "+Channel+".");
        Weight = W.getData();
        Bias = B.getData();
    }


    @Override
    public Tensor forward(Tensor input) {
        if (input.shape.length != 3){
            System.out.println("Error: input shape'length  must be 3.");
            return null;
        }
        if (input.shape[0] != Channel){
            System.out.println("Error: input shape'Channel   must be "+Channel);
            return null;
        }
        int channel = input.shape[0];
        int height = input.shape[1];
        int weight = input.shape[2] ;
        float[][][] inputDataRaw = input.getData3D();
        float[][][] inputData = new float[channel][height][weight];
        float[][][] outData = new float[channel][height][weight];
        for (int i = 0; i < channel; i++)   // 获取数据副本到 inputData
            for (int j = 0; j < height ; j++)
                System.arraycopy(inputDataRaw[i][j], 0, inputData[i][j],0, inputDataRaw[i][j].length);
        for (int i = 0; i < Channel; i++) {
            Tensor tempTensor = new Tensor(inputData[i]);    // 将该 2d 数据存储成Tensor，方便求mean与var
            float mean  = tempTensor.meanAllData();
            float var = tempTensor.varAllData();
            for (int h = 0; h < height ; h++) {             // 对每个元素 标准化
                for (int w = 0; w < weight ; w++) {
                    outData[i][h][w] = (float) ((float) (inputData[i][h][w] - mean)/(Math.pow(var + 0.00001,0.5)))*Weight[i] + Bias[i];
                }
            }


        }

        return new Tensor(outData);
    }

}
