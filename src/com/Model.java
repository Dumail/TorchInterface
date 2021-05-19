package com;

/*
网络模型，包含多层网络
 */
public class Model extends Module {

    public Module[] layers;

    public Model(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    public void setLayers(Module[] layers) {
        this.layers = layers;
    }

    public boolean loadModel(String path) {
        return true;
    }

    public Tensor forward(Tensor input) {
        Tensor tempTensor = input;
        for (Module layer : layers)
            tempTensor = layer.forward(tempTensor);
        return tempTensor;
    }
}
