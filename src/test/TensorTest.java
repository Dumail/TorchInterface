package test;

import com.Tensor;
import com.Util;
import org.junit.jupiter.api.AfterEach;

import java.util.Arrays;

class TensorTest {
    Tensor tensor;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        float[][] data = new float[][]{{1, 2, 3}, {4, 5, 6}};
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
    void sum() {
        System.out.println("Sum: " + tensor.sum());
    }

    @org.junit.jupiter.api.Test
    void dot() {
        System.out.println("Dot result: " + tensor.dot(new Tensor(new float[][]{{2, 3, 2}, {3, 2, 2}})));
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
    void unSqueeze() {
        boolean success = tensor.unSqueeze(0);
        success = tensor.unSqueeze(-1) && success;
        assert success;
    }

    @org.junit.jupiter.api.Test
    void squeeze() {
        tensor.unSqueeze(0);
        tensor.unSqueeze(-1);
        tensor.unSqueeze(2);
        tensor.squeeze();
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
    void getSlice() {
        Tensor tensor1D = new Tensor(new float[]{1, 2, 3, 4, 5});
        System.out.println("1d slice: " + tensor1D.getSlice(new int[]{-4, Integer.MAX_VALUE}));
        System.out.println("2d slice: " + tensor.getSlice(new int[]{0, 3}, new int[]{1, 3}));
    }

    @org.junit.jupiter.api.Test
    void equal() {
        Tensor tensorTest = new Tensor(new float[][]{{1, 2, 3}, {4, 5, 6}});
        System.out.println(tensorTest);
        System.out.println("Equal: " + tensor.equals(tensorTest));
    }

    @org.junit.jupiter.api.Test
    void getData2D() {
        float[][] data = tensor.getData2D();
        for (int i = 0; i < tensor.shape[0]; i++)
            System.out.println(Arrays.toString(data[i]));
    }

    @org.junit.jupiter.api.Test
    void getData() {
        Tensor tensorTest = new Tensor(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}});
        System.out.println("original : " + tensorTest);
        Tensor tensorGetData = tensorTest.getData("[:,:1]");
        System.out.println("result: " + tensorGetData);
    }

    @org.junit.jupiter.api.Test
    void T() {
        Tensor tensor3D = new Tensor(new float[][][]{{{1, 2, 3}, {1, 2, 3}}, {{1, 2, 3}, {1, 2, 3}}});
        System.out.println("Before: " + tensor3D);
        System.out.println("After: " + tensor3D.T());
    }

    @org.junit.jupiter.api.Test
    void generate() {
        System.out.println(Util.onesTensor(new int[]{3, 2}));
        System.out.println(Util.rangeTensor(-2, 8));
    }

    @org.junit.jupiter.api.Test
    void add() {
        tensor.add(new float[]{1, 3});
    }

    @org.junit.jupiter.api.Test
    void multiFloat() {
        tensor.multi(10);
    }
}