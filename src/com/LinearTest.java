package com;

import static org.junit.jupiter.api.Assertions.*;

class LinearTest {
    Linear linear;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        linear = new Linear(3, 3);
    }

    @org.junit.jupiter.api.AfterEach
    void tearDown() {
        System.out.println(linear);
    }

    @org.junit.jupiter.api.Test
    void forward() {
        Tensor input = new Tensor(new float[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor parameters = new Tensor(new int[]{4,3});
        System.out.println(linear.forward(input));
    }
}