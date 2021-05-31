package com;

/*
Relu Layer
 */
public class Relu extends Module {

    public Relu(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor tensor = new Tensor(input.shape);
        tensor.setData(Util.relu(input.getData()));
        return tensor;
    }

}
