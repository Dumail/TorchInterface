package test;

import com.BatchNorm2d;
import com.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BatchNorm2dTest {
    BatchNorm2d batchNorm2d;

    @BeforeEach
    void setUp() {
        batchNorm2d = new BatchNorm2d(2, 2, 2);
    }

    @AfterEach
    void tearDown() {
    }

    @Test
    void forward() {
        Tensor input = new Tensor(new float[][][]{{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {1, 2, 3}}});
        System.out.println("Input: " + input);
        System.out.println("BatchNorm2d Result: " + batchNorm2d.forward(input));
    }
}