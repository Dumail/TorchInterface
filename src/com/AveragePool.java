package com;

import java.util.Arrays;

/**
 * @program: TorchInterface
 * @description:
 * @author: R.
 * @create: 2021-05-31-23-59
 **/

public class AveragePool extends Pool {
    String paddings = "VALID";

    /**
     * @param args 二维数组，包含卷积核大小和可选参数（填充大小，步长），都使用数组进行传递，数组的第几个元素就表示第几维的参数
     */
    public AveragePool(int[]... args) {
        super(args);
        System.out.println("args is  " + Arrays.deepToString(args));
    }

    //    先根据输入feature的宽高和步长来计算输出的宽高,然后计算需要填充的尺寸,最后计算四个边分别填充的尺寸，填充顺序是优先填充右下，然后填充左上，可以非对称填充
//    padding = 'VALID' ,不做填充，不够卷积的边舍弃
//    https://zhuanlan.zhihu.com/p/102268312
    @Override
    public Tensor forward(Tensor input) {
        int channel = input.shape[0];
        int height = (input.shape[1] + 2 * this.padding[0] - kernelSize[0]) / this.step[0] + 1;
        int weight = (input.shape[2] + 2 * this.padding[1] - kernelSize[1]) / this.step[1] + 1;
        int inputHeight = input.shape[1] + this.padding[0] * 2;
        int inputWeight = input.shape[2] + this.padding[1] * 2;
        float[][][] inputDataRaw = input.getData3D();//原始输入数组

        float[][][] outData = new float[channel][height][weight];
        float[][][] inputData = new float[channel][inputHeight][inputWeight];
        for (int i = 0; i < channel; i++)
            for (int j = this.padding[0]; j < inputHeight; j++)
                System.arraycopy(inputDataRaw[i][j], 0, inputData[i][j], this.padding[1], inputDataRaw[i][j].length);

        for (int c = 0; c < channel; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < weight; w++) {
                    float sum = (float) 0.0;
                    int time = 0;
                    for (int kh = 0; kh < kernelSize[0]; kh++) {
                        for (int kw = 0; kw < kernelSize[1]; kw++) {
                            sum += inputData[c][h * step[0] + kh][w * this.step[1] + kw];
                            time++;
                        }
                    }
                    outData[c][h][w] = (float) sum / time;
                }
            }
        }

        Tensor outTensor = new Tensor(outData);
        outTensor.squeeze();
        return outTensor;

//        If padding == “SAME”: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
//        If padding == “VALID”: output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i])
//        https://www.imooc.com/article/73051/
//        在Tensorflow中池化（包括平均池化和最大池化）时如果有填充像素，是不考虑填充像素的，也就是说最大池化的时候如果有填充，实际上填充的是-inf（负无穷），-1当然比负无穷大咯。
//        Tensorflow中池化层如果有填充，不考虑填充像素的数值。
    }

    @Override
    public String toString() {
        return super.toString() + "-->AveragePool{" +
                "paddings='" + paddings + '\'' +
                '}';
    }
}