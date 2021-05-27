package com;

import static org.junit.jupiter.api.Assertions.*;

class LinearTest {
    Linear linear;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        linear = new Linear(2, 3);
    }

    @org.junit.jupiter.api.AfterEach
    void tearDown() {
        System.out.println(linear);
    }

    @org.junit.jupiter.api.Test
    void forward() {
        Tensor input = new Tensor(new float[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor parameters = new Tensor(4, 3);
        System.out.println(linear.forward(input));
    }

    @org.junit.jupiter.api.Test
    void readParameters() {
        assert linear.readParameters("src/com/pars.pt");
    }
}