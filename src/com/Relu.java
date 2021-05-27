package com;

/*
Relu激活函数
 */
public class Relu extends Layer {

    public Relu(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor tensor = new Tensor(input.shape) ;
        tensor.setData(Util.relu(input.getData()));
        return tensor;
    }


}
