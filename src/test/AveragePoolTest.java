package test;

import com.AveragePool;
import com.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AveragePoolTest {
    AveragePool averagePool;

    @BeforeEach
    void setUp() {
        averagePool = new AveragePool(new int[]{2, 2});
    }

    @AfterEach
    void tearDown() {
        System.out.println("AveragePool: " + averagePool);
    }

    @Test
    void forward() {
        Tensor input = new Tensor(new float[][][]{{{1, 2, 3}, {4, 5, 10}, {7, 8, 9}}, {{1, 2, 3}, {4, 5, 6}, {7, 1, 9}}});
        Tensor output = averagePool.forward(input);

        System.out.println("Input: " + input);
        System.out.println("Result: " + output);
    }
}