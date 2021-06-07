package com;

public class Main {

    static class MyNet extends Network {
        public MyNet(int inputSize, int outputSize) {
            super(inputSize, outputSize);
            this.modules = new Module[]{
                    new Conv2D(inputSize, 128, new int[]{4, 4}),
                    new BatchNorm2d(128),
                    new Relu(),
                    new Conv2D(128, 256, new int[]{3, 3}),
                    new BatchNorm2d(256),
                    new MaxPool(new int[]{3, 3}),
                    new Relu(),
                    new Linear(256, 128),
                    new Relu(),
                    new Linear(128, outputSize)

            };
        }
    }

    public static void main(String[] args) {
        MyNet myNet = new MyNet(121, 1);
        myNet.readParameters("src/test/predictor.pt");
        Tensor data = Util.rangeTensor(0, 121 * 8 * 8).reshape(new int[]{121, 8, 8});
        System.out.println(myNet.forward(data));
    }
}
