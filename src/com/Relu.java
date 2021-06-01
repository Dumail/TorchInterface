package com;

/*
Relu Layer
 */
public class Relu extends Module {

    public Relu() {
        super(0, 0);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor tensor = new Tensor(input.shape);
        tensor.setData(Util.relu(input.getData()));
        return tensor;
    }

    @Override
    public String toString() {
        return super.toString() + "-->Relu Module.";
    }
}
