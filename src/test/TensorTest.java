package test;

import com.Tensor;
import org.junit.jupiter.api.AfterEach;

class TensorTest {
    Tensor tensor;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        float[][] data = new float[][]{{1, 1, 1}, {1, 1, 1}};
        tensor = new Tensor(data);
    }

    @AfterEach
    void tearDown() {
        System.out.println(tensor);
    }

    @org.junit.jupiter.api.Test
    void randomData() {
        tensor.randomData();
    }

    @org.junit.jupiter.api.Test
    void reshape() {
        if (tensor.reshape(new int[]{3, 2}))
            System.out.println("Shape change successful.");
    }

    @org.junit.jupiter.api.Test
    void expand() {
        tensor.expand(new int[]{3, 4}, 0);
    }

    @org.junit.jupiter.api.Test
    void multi() {
        Tensor tensorTest = new Tensor(new float[][]{{1}, {1}, {1}});
        System.out.println(tensorTest);
        System.out.println("\nResult: " + tensor.multi(tensorTest) + "\n");
    }

    @org.junit.jupiter.api.Test
    void getOfIndex() {
        System.out.println(tensor.getOfIndex(0, 0));
        System.out.println(tensor.getOfIndex(1, 0));
        System.out.println(tensor.getOfIndex(1, 2));
    }

    @org.junit.jupiter.api.Test
    void equal() {
        Tensor tensorTest = new Tensor(new float[][]{{1, 2, 3}, {4, 5, 6}});
        System.out.println(tensorTest);
        System.out.println("Equal: " + tensor.equals(tensorTest));
    }

    @org.junit.jupiter.api.Test
    void getData() {
        Tensor tensorTest = new Tensor(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}});
        System.out.println("original : " + tensorTest);
        Tensor tensorGetData = tensorTest.getData("[:,:1]");
        System.out.println("result : " + tensorGetData);

    }


}