package test;

import com.Linear;
import com.Tensor;

import static org.junit.jupiter.api.Assertions.*;

class LinearTest {
    Linear linear;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        linear = new Linear(3, 2);
    }

    @org.junit.jupiter.api.AfterEach
    void tearDown() {
        System.out.println("Finally linear: " + linear);
    }

    @org.junit.jupiter.api.Test
    void forward() {
        Tensor input = new Tensor(new float[][]{{1, 2, 3}, {3, 4, 5}});
        Tensor parameters = new Tensor(3, 3);
        System.out.println("Forward result: " + linear.forward(input) + "\n");
    }

    @org.junit.jupiter.api.Test
    void readParameters() {
        assert linear.readParameters("src/com/test_pars.pt");
        Tensor input = new Tensor(new float[][]{{1, 1, 2}});
        System.out.println("Forward result: " + linear.forward(input) + "\n");
    }
}