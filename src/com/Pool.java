package com;

import java.util.Arrays;

public class Pool extends Layer {
    int[] kernelSize;//卷积核大小
    int[] padding = new int[]{0, 0}; //填充
    int[] step = new int[]{1, 1};//步长
    String paddingType = "zero"; //填充类型，补零

    /**
     * 池化层构造函数
     * <p>
     * 滤波形状大小如 [3,3] (一位数代表 [n,n]) 、  padding 情况 (默认VALID：0)、step 长度(默认1)
     */
    public Pool(int[]... args) {
        super(0, 0);

        if (args.length == 0) {
            System.out.println("Error: args must be  1 or 2." + Arrays.toString(args));
            return;
        }
        int dim = args[0].length;  //卷积维度
        if (dim == 1) {
            kernelSize = new int[]{args[0][0], args[0][0]};
        } else if (dim == 2) {
            kernelSize = Arrays.copyOf(args[0], dim);
        } else
            System.out.println("Error: args dim must be 1 or 2.");

        if (args.length > 1) {  // padding 0 代表 Valid ，1 代表 Same
            padding = new int[]{0, 0};
            if (args[1].length == 2) {
                padding[0] = args[1][0];
                padding[1] = args[1][1];
            }

        }

        if (args.length >= 2) {          //  存储步长，默认为
            if (args[2].length == 1)
                step = new int[]{args[2][0], args[2][0]};
            else if (args[2].length == 2)
                step =  Arrays.copyOf(args[2], args[2].length);
            else
                System.out.println("Error: args step dim must be 1 or 2.");
        }
    }

    @Override
    public Tensor forward(Tensor input) {

        return null;
    }
}

