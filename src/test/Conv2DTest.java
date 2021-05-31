package test;

import com.AveragePool;
import com.Conv2D;
import com.MaxPool;
import com.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
        AveragePool averagePool = new AveragePool(2, 2, shape);
        maxPool = new MaxPool(2, 2, shape);
        Tensor input = new Tensor(new float[][][]{{{1, 2, 5}, {4, 5, 10}, {7, 8, 9}},{{1, 2, 3}, {4, 5, 6}, {7, 1, 9}}});
        Tensor output = maxPool.forward(input);

        System.out.println("Result: " + input);
        System.out.println("Result: " + averagePool.forward(input));
        System.out.println("Result: " + output);


    }

}