package com;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/*
神经网络基本数据类型：张量
 */
public class Tensor {
    public int[] shape; //张量形状
    private float[] data; //张量数据，采取一维数组存储

    /**
     * 从数据形状构造张量
     *
     * @param shape 形状数组
     */
    public Tensor(int... shape) {
        this.shape = new int[shape.length];
        System.arraycopy(shape, 0, this.shape, 0, dims());
        this.data = new float[Util.prod(shape)];
    }

    /**
     * 从数据形状构造张量
     *
     * @param shapeList 形状列表
     */
    public Tensor(ArrayList<Integer> shapeList) {
        this.shape = new int[shapeList.size()];
        for (int i = 0; i < shapeList.size(); i++)
            shape[i] = shapeList.get(i);
        this.data = new float[Util.prod(shape)];
    }

    /**
     * 从另一个张量新建张量，两者指向同一数据
     *
     * @param tensor 输入张量
     */
    public Tensor(Tensor tensor) {
        this.data = tensor.data;
        this.shape = new int[tensor.shape.length];
        System.arraycopy(tensor.shape, 0, this.shape, 0, tensor.dims());
    }

    /**
     * 从一维数据构造张量
     *
     * @param data 数组数据
     */
    public Tensor(float[] data) {
        this.data = new float[data.length];
        System.arraycopy(data, 0, this.data, 0, data.length);
        this.shape = new int[]{data.length};
    }

    /**
     * 从二维数据构造张量
     *
     * @param data 数组数据
     */
    public Tensor(float[][] data) {
        this.data = new float[data.length * data[0].length];
        // 逐层复制数据
        for (int i = 0; i < data.length; i++)
            System.arraycopy(data[i], 0, this.data, i * data[0].length, data[0].length);
        this.shape = new int[]{data.length, data[0].length};
    }

    /**
     * 从三维数据构造张量
     *
     * @param data 数组数据
     */
    public Tensor(float[][][] data) {
        this.data = new float[data.length * data[0].length * data[0][0].length];
        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                System.arraycopy(data[i][j], 0, this.data, i * data[0].length * data[0][0].length + j * data[0][0].length, data[0][0].length);
        this.shape = new int[]{data.length, data[0].length, data[0][0].length};
    }

    /**
     * 随机化数据
     */
    public void randomData() {
        Random rand = new Random();
        for (int i = 0; i < data.length; i++)
            data[i] = rand.nextFloat();
    }

    /**
     * 将数据全设为1
     */
    public void onesData() {
        Arrays.fill(this.data, 1);
    }

    /**
     * 索引指定位置的值
     *
     * @param index 索引，需要与张量的维度一样
     * @return 指定值
     */
    public float getOfIndex(int... index) {
        //只能一次索引一个值，且不能省略0
        if (this.dims() != index.length) {
            System.out.println("Warning" + Util.getPos() + ": index shape " + this.dims() + " mismatch data shape " + index.length);
            return -1;
        }
        int pos = 0; //总的位置下标
        for (int i = 0; i < index.length - 1; i++)
            pos += index[i] * this.shape[i + 1];
        pos += index[index.length - 1];
        return this.data[pos];
    }

    /**
     * 通过指定的开始坐标和和结束坐标获取张量切片
     *
     * @param indices 一个数组代表一个维度，维度从高到低。 每个数组指定开始和结束的位置，负数表示从后往前，大于该维度总数代表取该维度所有数
     * @return 切片结果张量
     */
    public Tensor getSlice(int[]... indices) {
        //TODO 任意维度切片
        //二维切片
        if (this.dims() > 2) {
            System.out.println("Error" + Util.getPos() + " Current version does not support high dim.");
            return null;
        }

        //第一维起始点
        int rowStart = indices[0][0];
        int rowEnd = indices[0][1];
        //限定切片范围
        rowStart = rowStart < 0 ? this.shape[0] + rowStart : Math.min(rowStart, this.shape[0]);
        rowEnd = rowEnd < 0 ? this.shape[0] + rowEnd : Math.min(rowEnd, this.shape[0]);
        //处理后的值不能小于0
        if (rowStart < 0 || rowEnd < 0) {
            System.out.println("Error" + Util.getPos() + " index out of low bound.");
            return null;
        }
        if (rowStart > rowEnd) {
            System.out.println("Error" + Util.getPos() + " Start index must smaller than end index.");
            return null;
            //TODO 反向切片
        }

        //一维切片
        if (this.dims() == 1) {
            float[] tempData = new float[rowEnd - rowStart];
            System.arraycopy(this.data, rowStart, tempData, 0, rowEnd - rowStart);
            return new Tensor(tempData);
        }

        //第二维起始点
        int colStart = indices[1][0];
        int colEnd = indices[1][1];
        //限定切片范围
        colStart = colStart < 0 ? this.shape[1] + colStart : Math.min(colStart, this.shape[1]);
        colEnd = colEnd < 0 ? this.shape[1] + colEnd : Math.min(colEnd, this.shape[1]);
        //处理后的值不能小于0
        if (colStart < 0 || colEnd < 0) {
            System.out.println("Error" + Util.getPos() + " index out of low bound.");
            return null;
        }
        if (colStart > colEnd) {
            System.out.println("Error" + Util.getPos() + " Start index must smaller than end index.");
            return null;
        }

        float[][] tempData = new float[rowEnd - rowStart][colEnd - colStart];
        for (int i = rowStart; i < rowEnd; i++)
            System.arraycopy(this.data, i * this.shape[1] + colStart, tempData[i - rowStart], 0, colEnd - colStart);
        return new Tensor(tempData);
    }

    /**
     * 改变张量形状
     *
     * @param shape 新的数据形状，-1表示该维度形状自动计算，最多只能有一个-1
     * @return 是否改变成功
     */
    public Tensor reshape(final int[] shape) {
        int[] tempShape = Arrays.copyOf(shape, shape.length);
        int length = Util.prod(tempShape); //给定形状总长度
        for (int i = 0; i < tempShape.length; i++)
            if (tempShape[i] == -1) {
                //自动计算为给定的维度形状，这种情况下length为负
                length = -length;
                tempShape[i] = data.length / length;
            }

        if (Util.prod(tempShape) != data.length) {
            System.out.println("Error" + Util.getPos() + " Data size " + data.length + " can not reshape to " + Arrays.toString(shape) + "!");
            return null;
        }

        Tensor tempTensor = new Tensor(this);
        tempTensor.shape = tempShape;
        return tempTensor;
    }

    /**
     * 转置
     *
     * @return 转置后的张量
     */
    public Tensor T() {
        int length1 = this.shape[dims() - 1];//最后一维形状
        int length2 = this.shape[dims() - 2];//倒数第二维形状
        int otherLength = Util.prod(this.shape) / length1 / length2;//其他维的形状
        float[] tempData = new float[otherLength * length2 * length1];
        for (int n = 0; n < otherLength; n++)
            //对于高纬度保持不变
            for (int i = 0; i < length2; i++)
                for (int j = 0; j < length1; j++) {
                    tempData[n * length1 * length2 + j * length2 + i] = this.data[n * length1 * length2 + i * length1 + j];
                }
        int[] tempShape = Arrays.copyOf(this.shape, this.shape.length);
        tempShape[dims() - 2] = length1;
        tempShape[dims() - 1] = length2;
        Tensor tempTensor = new Tensor(tempData);
        tempTensor = tempTensor.reshape(tempShape);
        return tempTensor;
    }

    /**
     * 去掉张量中形状为1的维度
     * 例如Ax1xBx1的张量处理后为AxB
     */
    public void squeeze() {
        //获取形状不是1的维度个数
        int tempDim = 0;
        for (int i = 0; i < dims(); i++)
            if (this.shape[i] != 1)
                tempDim++;
        int[] tempShape = new int[tempDim];

        //赋值
        int index = 0;
        for (int i = 0; i < dims(); i++)
            if (this.shape[i] != 1)
                tempShape[index++] = this.shape[i];
        this.shape = tempShape;
    }

    /**
     * 在指定位置上增加维度
     *
     * @param index 指定位置
     * @return 是否成功
     */
    public boolean unSqueeze(int index) {
        index = index < 0 ? dims() + index + 1 : Math.min(index, this.shape[1]);
        if (index < 0 || index > dims()) {
            System.out.println("Error" + Util.getPos() + " index must bigger than 0 and smaller than " + dims());
            return false;
        }
        int[] tempShape = new int[dims() + 1];
        int i = 0;
        for (; i < index; i++)
            tempShape[i] = this.shape[i];
        tempShape[i] = 1; //在指定位置添加一个维度
        for (; i < dims(); i++)
            tempShape[i + 1] = this.shape[i];
        this.shape = tempShape;
        return true;
    }

    /**
     * 将二维张量扩展为指定形状，向索引增加的方向进行扩展，扩展的部分填充指定数据
     *
     * @param shape 扩展后形状
     * @param pad   填充数据
     * @return 是否扩展成功
     */
    public boolean expand(int[] shape, float pad) {
        if (dims() != this.dims()) {
            System.out.println("Error" + Util.getPos() + " Shape dimension " + this.dims() + " mismatch " + dims());
            return false;
        }
        for (int i = 0; i < dims(); i++) {
            if (this.shape[i] > shape[i]) {
                System.out.println("Error" + Util.getPos() + " Shape " + Arrays.toString(this.shape) + " can not be less than " + Arrays.toString(shape));
                return false;
            }
        }

        //二维张量扩展，先分配空间，然后依次复制过去
        float[] tempData = new float[Util.prod(shape)];
        //列扩展
        for (int i = 0; i < this.shape[0]; i++) {
            System.arraycopy(this.data, i * this.shape[1], tempData, i * shape[1], this.shape[1]);
            for (int j = this.shape[1]; j < shape[1]; j++)
                tempData[i * shape[1] + j] = pad;
        }
        for (int i = this.shape[0]; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                tempData[i * shape[1] + j] = pad;
        //行扩展
        this.data = tempData;
        System.arraycopy(shape, 0, this.shape, 0, dims());
        return true;
    }

    /**
     * 求张量所有元素之和
     *
     * @return 求和结果
     */
    public float sum() {
        float sum = 0;
        for (float datum : this.data) sum += datum;
        return sum;
    }

    /**
     * 求张量数据的均值
     *
     * @return 均值
     */
    public float meanAllData() {
        return sum() / data.length;
    }

    /**
     * 求张量数据的方差
     *
     * @return 方差
     */
    public float varAllData() {
        float var = 0;
        float mean = meanAllData();
        for (float datum : this.data) var += Math.pow((datum - mean), 2);
        return var / data.length;
    }


    /**
     * 求两个张量的点乘
     *
     * @param input 另一个张量
     * @return 点乘结果，出错将返回最大浮点数
     */
    public float dot(Tensor input) {
        if (!Arrays.equals(this.shape, input.shape)) {
            System.out.println("Error" + Util.getPos() + " shape " + Arrays.toString(this.shape) + " mismatch input shape " + Arrays.toString(input.shape));
            return Float.MAX_VALUE;
        }
        float[] tempData = input.getData();
        float sum = 0;
        for (int i = 0; i < this.data.length; i++)
            sum += tempData[i] * this.data[i];
        return sum;
    }

    /**
     * 矩阵乘法，将张量的最后两维与另一个张量的最后两维相乘
     *
     * @param input 另一个张量
     * @return 乘法结果
     */
    public Tensor multi(Tensor input) {
        //相乘张量至少有两维
        if (this.dims() < 2 || input.dims() < 2) {
            System.out.println("Error" + Util.getPos() + " Tensor dimension is too low.");
        }
        //两个张量维度需要相同
        if (this.dims() != input.dims()) {
            System.out.println("Error" + Util.getPos() + " Dimension " + this.dims() + " mismatch " + input.dims());
            return null;
        }

        //高于二维的形状需要相同
        for (int i = 0; i < this.dims() - 2; i++) {
            if (this.shape[i] != input.shape[i]) {
                System.out.println("Error" + Util.getPos() + " Multiplied this.shape " + Arrays.toString(shape) + " by this.shape " + Arrays.toString(input.shape));
                return null;
            }
        }

        //相乘矩阵的行和列
        int rowOfMat1 = this.shape[this.dims() - 2];
        int colOfMat1 = this.shape[this.dims() - 1];
        int rowOfMat2 = input.shape[input.dims() - 2];
        int colOfMat2 = input.shape[input.dims() - 1];

        //相乘的两维行与列相同
        if (colOfMat1 != rowOfMat2) {
            System.out.println("Error" + Util.getPos() + " Multiplied this.shape " + Arrays.toString(shape) + " by this.shape " + Arrays.toString(input.shape));
            return null;
        }

        //实例化输出张量
        int[] outShape = Arrays.copyOf(this.shape, this.dims());
        outShape[outShape.length - 1] = colOfMat2;
        float[] output = new float[Util.prod(outShape)];

        //TODO 任意维度张量相乘
        //二维矩阵相乘
        for (int i = 0; i < rowOfMat1; i++)
            for (int j = 0; j < colOfMat2; j++) {
                output[i * colOfMat2 + j] = 0;
                for (int k = 0; k < colOfMat1; k++)
                    output[i * colOfMat2 + j] += this.data[i * colOfMat1 + k] * input.data[k * colOfMat2 + j];
            }
        Tensor result = new Tensor(output);
        return result.reshape(new int[]{rowOfMat1, colOfMat2});
    }

    public int size() {   // 获取数据的长度
        return this.data.length;
    }

    public int dims() {
        return this.shape.length;
    }

    public void setData(float[] inputData) {
        System.arraycopy(inputData, 0, this.data, 0, inputData.length);
    }

    public void setShape(int[] shape) {
        System.arraycopy(shape, 0, this.shape, 0, shape.length);
    }

    public int[] getShape() {
        return Arrays.copyOf(this.shape, this.shape.length);
    }

    /**
     * 获取原始的数据
     *
     * @return 一维的原始数据
     */
    public float[] getData() {
        return Arrays.copyOf(this.data, this.data.length);
    }

    /**
     * 获取二维张量的数据
     *
     * @return 二维数组
     */
    public float[][] getData2D() {
        if (this.dims() != 2) {
            System.out.println("Error" + Util.getPos() + " Tensor dim is not 2.");
            return null;
        }
        float[][] tempData = new float[this.shape[0]][this.shape[1]];
        for (int i = 0; i < this.shape[0]; i++)
            System.arraycopy(this.data, i * this.shape[1], tempData[i], 0, this.shape[1]);
        return tempData;
    }

    /**
     * 获取三维张量的数据
     *
     * @return 三维数组
     */
    public float[][][] getData3D() {
        if (this.dims() != 3) {
            System.out.println("Error" + Util.getPos() + " Tensor dim is not 3");
            return null;
        }
        float[][][] tempData = new float[this.shape[0]][this.shape[1]][this.shape[2]];
        for (int i = 0; i < this.shape[0]; i++)
            for (int j = 0; j < this.shape[1]; j++)
                System.arraycopy(this.data, i * this.shape[1] * this.shape[2] + j * this.shape[2], tempData[i][j], 0, this.shape[2]);
        return tempData;
    }

    /**
     * 获取四维张量的数据
     *
     * @return 四维数组
     */
    public float[][][][] getData4D() {
        if (this.dims() != 4) {
            System.out.println("Error" + Util.getPos() + " Tensor dim is not 4");
            return null;
        }
        float[][][][] tempData = new float[this.shape[0]][this.shape[1]][this.shape[2]][this.shape[3]];
        for (int i = 0; i < this.shape[0]; i++)
            for (int j = 0; j < this.shape[1]; j++)
                for (int k = 0; k < this.shape[2]; k++)
                    System.arraycopy(this.data, i * this.shape[1] * this.shape[2] * this.shape[3] + j * this.shape[2] * this.shape[3] + k * this.shape[3], tempData[i][j][k], 0, this.shape[3]);
        return tempData;
    }

    //TODO complete
    public Tensor getData(String inputs) {
        // 模仿 python 列表切片的输入 , 暂不实现 -1 操作 ,也不包括 只取一行的操作
        if (!inputs.startsWith("[") || !inputs.endsWith("]") || inputs.split(",").length != dims())
            return null;
        String[] splits = inputs.replace("[", "").replace("]", "").split(",");
        int[][] newShape = new int[dims()][2];
        int[] rsShape = new int[dims()];
        for (int i = 0; i < dims(); i++) {
            int originalLength = shape[i];
            String indexSplit = splits[i];
            int start = 0;
            int end = originalLength;
            if (indexSplit.equals(":")) {
                newShape[i][0] = 0;
                newShape[i][1] = originalLength;
            } else if (indexSplit.length() == 1) {             //
                newShape[i][0] = 0;
                newShape[i][1] = originalLength;
            } else if (indexSplit.split(":").length == 1) {
                start = Integer.parseInt(indexSplit.split(":")[0]);
                newShape[i][0] = start;
                newShape[i][1] = end;
            } else if (indexSplit.split(":").length == 2) {
                start = indexSplit.split(":")[0].length() != 0 ? Integer.parseInt(indexSplit.split(":")[0]) : 0;
                newShape[i][0] = start;
                end = indexSplit.split(":")[1].length() != 0 ? Integer.parseInt(indexSplit.split(":")[1]) : originalLength;
                newShape[i][1] = end;
            }
            if (end > originalLength || start > originalLength || start >= end || start < 0) {
                return null;  //切片输入不合法
            }
            rsShape[i] = newShape[i][1] - newShape[i][0];
        }   // of for

        int[] shapeSize = new int[dims() + 1];   // 用来保存每一维包含的元素个数
        int size = 1;
        shapeSize[dims()] = 1;
        List<Float> lists = new ArrayList<>();
        System.arraycopy(shape, 0, shapeSize, 0, rsShape.length);
        int[] tmpSize = new int[shapeSize.length];
        System.arraycopy(shapeSize, 0, tmpSize, 0, shapeSize.length);
        for (int i = 0; i < dims(); i++) {        // 计算每个维度所含的元素个数
            size = size * tmpSize[dims() - i];
            shapeSize[dims() - 1 - i] = size;
        }// of for
        for (int i = 0; i < data.length; i++) {        // 遍历明日歌元素，从index计算出其位置，判断是否符合要求的下标
            boolean isAdd = true;
            int start = i;      // 记录开始位置
            for (int j = 0; j < dims(); j++) {
                int index = start / shapeSize[j];         // 下标除以第 j 维的元素个数，可以得到该元素在第 j 维的位置
                if (index < newShape[j][0] || index >= newShape[j][1]) {   // 判断其位置是否在输入的对应维度区间内
                    isAdd = false;
                    break;
                }
                start = start % shapeSize[j];              // 之后开始判断下一维的位置，从高维到低维
            } // of for j

            if (!isAdd)
                continue;
            lists.add(data[i]);
        } // of for  i

        // 目前是返回一个新Tensor
        Tensor tensor = new Tensor(rsShape);
        int len = lists.size();
        float[] rsData = new float[len];
        for (int i = 0; i < len; i++) {
            rsData[i] = lists.get(i);
        }
        tensor.setData(rsData);

        // 修改自身数据
//        int len = lists.size();
//        float[] rsData = new float[len];
//        for (int i = 0; i < len ; i++) {
//            rsData[i] = lists.get(i);
//        }
//        shape = rsShape ;
//        setData(rsData);

        return tensor;
    }

    @Override
    public String toString() {
        return "Tensor : \n shape: " + Arrays.toString(this.shape) + " \n" + " data: " + Arrays.toString(this.data);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tensor tensor = (Tensor) o;
        return Arrays.equals(shape, tensor.shape) && Arrays.equals(data, tensor.data);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

    /**
     * 广播操作，对所有元素执行同一操作
     *
     * @param func 执行的函数
     */
    public void broadcast(Function<Float, Float> func) {
        for (int i = 0; i < this.data.length; i++)
            this.data[i] = func.apply(this.data[i]);
    }

    /**
     * 广播加法，张量的所有数据都加上一个数
     *
     * @param value 加数
     */
    public void add(float value) {
        broadcast((x) -> x + value);
    }

    /**
     * 广播操作，张量所有数据乘一个数
     *
     * @param value 乘数
     */
    public void multi(float value) {
        broadcast((x) -> x * value);
    }

    /**
     * 广播加法，张量加上一个浮点数组，则第一维中每个张量的数据都加上数组中对应的数
     *
     * @param value 加数数组
     */
    public void add(float[] value) {
        if (value.length != this.shape[0])
            System.out.println("Error" + Util.getPos() + "length of input " + value.length + " mismatch tensor shape " + Arrays.toString(this.shape) + " for adding.");
        int otherLength = Util.prod(this.shape) / this.shape[0]; //除了第一维的其他维长度
        for (int i = 0; i < value.length; i++)
            for (int j = 0; j < otherLength; j++)
                this.data[i * otherLength + j] += value[i];
    }


//    @Override
//    public Tensor clone()
//    {
//        Tensor clone = (Tensor) super.clone();
//        Tensor outTensor = new Tensor(this.data);
//        outTensor.setShape(shape);
//        return outTensor;
//    }

    /**
     * 复制该张量，开辟新的空间
     */
    @Override
    public Tensor clone() {
        Tensor cloneTensor = null;
        try {
            cloneTensor = (Tensor) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        assert cloneTensor != null;
        cloneTensor.setData(this.data);
        cloneTensor.setShape(this.shape);
        return cloneTensor;
    }
}
