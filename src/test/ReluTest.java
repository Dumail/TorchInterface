package test;

import com.Relu;
import com.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class ReluTest {
    Relu relu;

    @BeforeEach
    void setUp() {
        relu = new Relu();
    }

    @AfterEach
    void tearDown() {
        System.out.println(relu);
    }

    @Test
    void forward() {
        Tensor input = new Tensor(new float[][]{{1, -1.0f, 0.0f}, {4, 5.2f, -6}});
        System.out.println("Input: " + input);
        System.out.println("Result: " + relu.forward(input));
    }
}