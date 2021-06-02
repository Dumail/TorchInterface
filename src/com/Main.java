package com;

public class Main {

    static class MyNet extends Network {
        public MyNet(int inputSize, int outputSize) {
            super(inputSize, outputSize);
            this.modules = new Module[]{
                    new Conv2D(inputSize, 4, new int[]{2, 2}),
                    new Relu(),
                    new Conv2D(4, 4, new int[]{2, 2}),
                    new MaxPool(new int[]{3, 3}),
                    new Relu(),
                    new Linear(4, 2),
                    new Relu(),
                    new Linear(2, outputSize),
            };
        }
    }

    public static void main(String[] args) {
        MyNet net = new MyNet(4, 1);
        Tensor input = Util.rangeTensor(0, 4 * 5 * 5);
        if (input.reshape(new int[]{4, 5, 5}))
            System.out.println(input);
        System.out.println(net.forward(input));
    }
}
