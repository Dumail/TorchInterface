package com;

public class Main {

    class MyNet extends Network {
        public MyNet(int inputSize, int outputSize) {
            super(inputSize, outputSize);
            this.modules = new Module[]{
                    new Conv2D(3, 3, new int[]{3, 3}),
                    new Relu(),
                    new Conv2D(3, 3, new int[]{3, 3}),
                    new MaxPool(new int[]{3, 3}),
                    new Relu(),
                    new Linear(16, 8),
                    new Relu(),
                    new Linear(8, 1),
            };
        }
    }

    public static void main(String[] args) {
    }
}
