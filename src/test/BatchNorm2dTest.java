package test;

import com.BatchNorm2d;
import com.Tensor;
import com.Util;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BatchNorm2dTest {
    BatchNorm2d batchNorm2d;

    @BeforeEach
    void setUp() {
        batchNorm2d = new BatchNorm2d(4);
    }

    @AfterEach
    void tearDown() {
        System.out.println(batchNorm2d);
    }

    @Test
    void readParameters() {
        batchNorm2d.readParameters("src/test/test_pars_batchNorm.pt");
    }

    @Test
    void forward() {
        Tensor input = Util.rangeTensor(0, 24).reshape(new int[]{4, 2, 3});
        System.out.println("Input: " + input);
        System.out.println("BatchNorm2d Result: " + batchNorm2d.forward(input));
    }
}