package test;

import com.Conv2D;
import com.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Conv2DTest {

    Conv2D conv;

    @BeforeEach
    void setUp() {
        conv = new Conv2D(1, 2, new int[]{2, 2});
    }

    @AfterEach
    void tearDown() {
        System.out.println(conv);
    }

    @Test
    void forward() {
        Tensor input = new Tensor(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        Tensor output = conv.forward(input);
        System.out.println("Result: " + output);
    }
}