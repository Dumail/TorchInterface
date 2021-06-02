package test;

import com.Module;
import com.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;


class NetworkTest {
    /**
     * 自定义网络结构
     */
    static class selfNetwork extends Network {
        public selfNetwork(int inputSize, int outputSize) {
            super(inputSize, outputSize);

            //将需要的网络层放入
            this.modules = new Module[]{
                    new Conv2D(inputSize, 3, new int[]{2, 2}),
                    new Linear(12, 2),
            };
        }
    }

    Network net;

    @BeforeEach
    void setUp() {
        net = new selfNetwork(2, 2);
    }

    @AfterEach
    void tearDown() {
        System.out.println(net);
    }

    @Test
    void readParameters() {
        assert net.readParameters("src/test/test_pars_network.pt");
    }

    @Test
    void forward() {
        Tensor input = new Tensor(new float[][][]{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}});
        System.out.println("Result: " + net.forward(input));
    }
}