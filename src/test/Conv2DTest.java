package test;

import com.Conv2D;
import com.Tensor;
import com.Util;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
        Tensor input = Util.onesTensor(new int[]{2, 3, 3});
        conv.setParameters(Util.onesTensor(new int[]{2, 2, 2, 2}));
        Tensor output = conv.forward(input);
        System.out.println("Result: " + output);
    }

    @Test
    void readParameters() {
        assert conv.readParameters("src/test/test_pars_conv2D.pt");
        Tensor input = Util.onesTensor(new int[]{2, 3, 3});
        System.out.println(input);
        Tensor output = conv.forward(input);
        System.out.println("Result: " + output);
    }
}
