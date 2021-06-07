package test;

import com.Module;
import com.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by Chaofan Pan on  上午10:07
 **/
public class MnistTest {
    static class MyNet extends Network {
        public MyNet(int inputSize, int outputSize) {
            super(inputSize, outputSize);
            this.modules = new Module[]{
                    new Conv2D(inputSize, 2, new int[]{3, 3}),
                    new BatchNorm2d(2),
                    new Relu(),
                    new Conv2D(2, 4, new int[]{3, 3}),
                    new BatchNorm2d(4),
                    new Relu(),
                    new Conv2D(4, 8, new int[]{3, 3}),
                    new BatchNorm2d(8),
                    new MaxPool(new int[]{5, 5}),
                    new Relu(),
                    new Linear(128, 64),
                    new Relu(),
                    new Linear(64, outputSize),
            };
        }
    }

    MyNet net;

    @BeforeEach
    void setUp() {
        net = new MyNet(1, 1);
        net.readParameters("src/test/test_pars_mnist.pt");
    }

    @Test
    void MainTest() {
        ArrayList<String> arrayList = new ArrayList<>();
        try {
            FileReader fr = new FileReader("src/test/mnist_test.csv");
            BufferedReader bf = new BufferedReader(fr);
            String str;
            // 按行读取字符串
            while ((str = bf.readLine()) != null) {
                arrayList.add(str);
            }
            bf.close();
            fr.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        String[] data_str = arrayList.get(0).split(",");
        float[] data = new float[data_str.length - 1];
        for (int i = 1; i < data_str.length; i++) {
            data[i - 1] = Float.parseFloat(data_str[i]);
        }
        float label = Float.parseFloat(data_str[0]);
        Tensor tensor_data = new Tensor(data).reshape(new int[]{1, 28, 28});
        System.out.println(tensor_data);
        System.out.println("Label: " + label + ", pred: " + net.forward(tensor_data));
    }
}
