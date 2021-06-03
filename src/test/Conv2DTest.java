package test;

import com.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class Conv2DTest {

    Conv2D conv;


    @BeforeEach
    void setUp() {
        conv = new Conv2D(2, 2, new int[]{2, 2});
    }

    @AfterEach
    void tearDown() {
        System.out.println(conv);
    }

    @Test
    void forward() {
        Tensor input = new Tensor(new float[][][]{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}});
        Tensor output = conv.forward(input);
        System.out.println("Result: " + output);
    }




    @Test
    void readParameters(){
        assert conv.readParameters("src/test/test_pars_conv2D.pt");
    }

    @Test
    void PoolTest(){   //

        MaxPool maxPool;

        int[] shape = new int[]{2, 2};
        AveragePool averagePool = new AveragePool(shape);
        maxPool = new MaxPool(shape);
        Tensor input = new Tensor(new float[][][]{{{1, 2, 3,4}, {4, 5, 6,7}, {7, 8, 9,10}},{{1, 2, 3,4}, {4, 5, 6,7}, {7, 8, 9,10}}});
        Tensor output = maxPool.forward(input);
        System.out.println("***********************************************************");
        System.out.println("Result: " + input);
        System.out.println("aver Result: " + averagePool.forward(input));
        System.out.println("max Result: " + output);
        System.out.println("***********************************************************");

    }
    @Test
    void BatchNorm2dTest(){   //

        BatchNorm2d batchNorm2d;
        batchNorm2d = new BatchNorm2d(2, 2,2);
        Tensor input = new Tensor(new float[][][]{{{1, 2, 3}, {4, 5, 6}},{{1, 2, 3}, {1, 2, 3}}});
        System.out.println("BatchNorm2d Result: " + batchNorm2d.forward(input));
        System.out.println("BatchNorm2d Result: " + Arrays.toString( batchNorm2d.forward(input).shape));



    }

}