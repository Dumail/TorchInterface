package com;

import java.util.Arrays;

public class demotest {

    public static void main(String[] args) {
//        int[] arr1 = new int[]{1,2};
//        int[] arr2 = new int[]{3,4};
//        int[] arr3 = new int[]{3};
//        testFunc(arr1,arr2,arr3,arr2);



        BatchNorm2d batchNorm2d;
        batchNorm2d = new BatchNorm2d(2, 2,2);
        Tensor input = new Tensor(new float[][][]{{{1, 2, 3}, {4, 5, 6}},{{1, 2, 3}, {1, 2, 3}}});
        System.out.println("Result: " + batchNorm2d.forward(input));
        System.out.println("Result: " +Arrays.toString( batchNorm2d.forward(input).shape));

//        MaxPool maxPool;
//
//        int[] shape = new int[]{2, 2};
//        AveragePool averagePool = new AveragePool(2, 2, shape);
//        maxPool = new MaxPool(2, 2, shape);
//        Tensor input = new Tensor(new float[][][]{{{1, 2, 5}, {4, 5, 10}, {7, 8, 9}},{{1, 2, 3}, {4, 5, 6}, {7, 1, 9}}});
//        Tensor output = maxPool.forward(input);
//
//        System.out.println("Result: " + input);
//        System.out.println("Result: " + averagePool.forward(input));
//        System.out.println("Result: " + output);

    }

    private static void testFunc(int[] arr1, int[] arr2, int[]... args) {
        System.out.println("original : " + Arrays.toString(arr1));
        System.out.println("original : " + Arrays.toString(arr2));
        System.out.println("original : " +args.length);
        System.out.println("original : " + Arrays.toString(args[0]));
        System.out.println("original : " + args[0]);
        System.out.println("original : " + args[0][0]);
        System.out.println("original : " + Arrays.toString(args[1]));
    }




}
