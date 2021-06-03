package com;


/**
 * @program: TorchInterface
 * @description: 2d 标准化 ,  参考博客 ： https://blog.csdn.net/weixin_44278406/article/details/105554268
 * @author: R.
 * @create: 2021-06-03-11-00
 **/

public class BatchNorm2d  extends  BatchNormal{
    /**
     * 构造函数
     *
     * @param args 第一个参数为 通道个数C，W 默认为1 ，B 默认为0
     */
    public BatchNorm2d(int... args) {
        super(args); //输出节点个数默认与 input 相同

        if (args.length != 1)
            System.out.println("Error: BatchNorm2d args must be 1." + args.length);
        channel = args[0];
        parameters = Util.onesTensor(new int[]{channel});
        bias = new float[channel];
    }

    public void initWB(Tensor W,Tensor B) {      // 初始化 偏移量W与附加量B ，若有该参数则调用该初始化方法，没有则不用。
        if (parameters.size() != channel || bias.length != channel)
            System.out.println("Error: BatchNorm2d Weight and Bias must be " + channel + ".");
        parameters = W.clone();
        bias = B.getData();
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.shape.length != 3) {
            System.out.println("Error: input shape's length  must be 3.");
            return null;
        }
        if (input.shape[0] != channel) {
            System.out.println("Error: input shape's Channel   must be " + channel);
            return null;
        }
        int channel = input.shape[0];
        int height = input.shape[1];
        int weight = input.shape[2];
        float[][][] inputDataRaw = input.getData3D();
        float[][][] inputData = new float[channel][height][weight];
        float[][][] outData = new float[channel][height][weight];
        for (int i = 0; i < channel; i++)   // 获取数据副本到 inputData
            for (int j = 0; j < height; j++)
                System.arraycopy(inputDataRaw[i][j], 0, inputData[i][j], 0, inputDataRaw[i][j].length);
        float[] weights = parameters.getData();
        for (int i = 0; i < this.channel; i++) {
            Tensor tempTensor = new Tensor(inputData[i]);    // 将该 2d 数据存储成Tensor，方便求mean与var
            float mean = tempTensor.meanAllData();
            float var = tempTensor.varAllData();
            for (int h = 0; h < height; h++) {             // 对每个元素 标准化
                for (int w = 0; w < weight; w++) {
                    outData[i][h][w] = (float) ((inputData[i][h][w] - mean) / (Math.pow(var + 0.00001, 0.5))) * weights[i] + bias[i];
                }
            }
        }
        return new Tensor(outData);
    }

    @Override
    protected boolean loadParameters(ParametersTuple tuple) {
        int[] weight_shape = tuple.getWeight_shape();
        int[] bias_shape = tuple.getBias_shape();
        float[] weight = tuple.getWeight();
        float[] bias = tuple.getBias();

        if (weight_shape.length != 1 || bias_shape.length != 1) {
            System.out.println("Error" + Util.getPos() + " Weight of batch norm must be 1 dimension, but this weight dimension is " + weight_shape.length);
            return false;
        }

        if (weight_shape[0] != channel) {
            System.out.println("Error" + Util.getPos() + " Weight number of this file " + bias_shape[0] + " mismatch channel " + channel);
            return false;
        }

        if (bias_shape[0] != channel) {
            System.out.println("Error" + Util.getPos() + " Bias number of this file " + bias_shape[0] + " mismatch channel " + channel);
            return false;
        }

        //复制参数
        float[] weightData = new float[weight_shape[0]];
        System.arraycopy(weight, 0, weightData, 0, weight_shape[0]);
        System.arraycopy(bias, 0, this.bias, 0, bias_shape[0]);

        this.bias = bias;
        setParameters(weightData);
        return true;
    }
}
