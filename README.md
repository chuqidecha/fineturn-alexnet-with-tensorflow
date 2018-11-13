# 使用TensorFlow微调AlexNet

## 从caffemode中获取预训练权值
TensorFlow中没有预训练好的AlexNet模型，利用[caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)工具可以将
在caffe上预训练好的AlexNet模型转成numpy的npy格式。该项目已经一年多没有人维护了，可能存在python、protobuf、tensorflow等版本不
兼容的问题。[这里](https://github.com/chuqidecha/caffe-tensorflow)是我改好的一个版本，使用Python3.6、protobuf3.6、tensorflow1.10版本。
从[Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)中可以下载在ImageNet上预训练好的AlexNet模型。
在[这里](https://pan.baidu.com/s/1b9N-z-5fYibKd8O2Vlg0Tg)下载已经转换好的参数文件(bvlc_alexnet.npy)。

## AlexNet模型结构与参数
AlexNet模型共有5个卷积层，3个全连接层，前两个卷积层和第五个卷积层后有池化层。
![image](./resources/alexnet.png)

1. 卷基层1
输入图像大小为227\*227\*3(BGR)；该层有96（每个GPU48个）个大小为11\*11\*3的卷积核，步长为4，不使用填充。
因此输出特征图大小为55\*55\*96（(227-11)/4+1=55）。参数个数为96\*11\*11\*3+96=34944。
卷基层1之后紧跟一个LRN层，输出大小不变。
2. 池化层1
大小为3\*3，步长为2。因此输出特征图大小为27\*27\*96（(55-3)/2+1=27）
3. 卷基层2
256个大小为5\*5\*48的卷积核（每个GPU各128个，分别作用于池化层1输出的前后48个通道），步长为1，使用填充。
因此输出特征图大小为2个27\*27\*128。参数个数为2\*(128\*5\*5\*48+128)=307456。卷基层2之后也会紧跟一个LRN层。
4. 池化层2
大小为3\*3,步长为2。因此输出特征图大小为2个13\*13\*128（(27-3)/2+1=13）。
5. 卷基层3
有384个大小为3\*3\*256的卷积核（每个GPU各192个，作用于池化层2的所有输出），步长为1，使用填充。
因此输出特征图大小为2个13\*13\*192。参数个数为2\*(192\*3\*3\*256+192)=885120。
6. 卷基层4
有384个大小为3\*3\*192的卷积核（仅作用于当前GPU），步长为1，使用填充。
因此输出特征图大小为2个13\*13\*192。参数个数为2\*(192\*3\*3\*192+192)=663936。
7. 卷基层5
有256个大小为3\*3\*192的卷积核（仅作用于当前GPU），步长为1，使用填充。
因此输出特征图大小为2个13\*13\*128。参数个数为2\*(128\*3*3\*192+128)=442624。
8. 池化层5
大小为3\*3，步长为2。因此输出特征图大小为2个6\*6\*128。
9. 全连接层1
节点数为4096，参数个数为6\*6\*128\*2\*4096+4096=37752832
10. 全连接层2
节点数为4096，参数个数为4096\*4096+4096=16781312
11. 输出层
节点数为1000，参数个数为4096\*1000+1000=4097000


## 在TensorFlow上实现AlexNet

### 环境
* tensorflow 1.10
* opencv 3.4.3.18

### 主要代码
- [src/inference.py](./src/inference.py)
    - inference AlexNet模型实现
    - load_weights_biases 从预训练模型中加载参数
- [src/validate_alexnet_on_imagenet.py](./src/validate_alexnet_on_imagenet.py) 验证模型是否正确
- [src/setting](./src/setting) 主要参数配置
-



### 验证模型
为了测试模型是否正确，并且参数是否被正确赋值，可以创建一个ImageNet原始模型（最后一层有1000个类别）并将微调的网络层设置为空。
从原始ImageNet数据集中随机抽取了几张图片进行预测分类，下面是分类结果：
![image](./resources/validate.png)
从上图可以看出，模型正确并且参数被正确赋值。

## 踩过的坑

### tf.get_variable重用变量时并不能修改trainable属性
```
import tensorflow as tf

if __name__ == '__main__':
    with tf.variable_scope("test"):
        weights = tf.get_variable("weights", shape=[10], initializer=tf.truncated_normal_initializer(stddev=0.1)) # 变量可训练的
        baises = tf.get_variable("baises", initializer=tf.constant(1.0),trainable=False) # 变量不可训练

    with tf.variable_scope("test",reuse=True):
        weights = tf.get_variable("weights", shape=[10], initializer=tf.truncated_normal_initializer(stddev=0.1),trainable=False)
        baises = tf.get_variable("baises", initializer=tf.constant(1.0),trainable=True)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(tf.trainable_variables())
```

> [output]:[<tf.Variable 'test/weights:0' shape=(10,) dtype=float32_ref>]

trainable集合中中的变量并没有发生变化。因此优化阶段调用Optimizer的minimize方法必须显示的给出var_list的参数，而不能使用默认的trainable集合。

```
    with tf.name_scope('train'):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

        # tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,var_list=gradients, global_step=global_step)
```

### global_step参数的更新
只有调用Optimizer的minimize方法完成参数更新之后global\_step参数才会+1， 学习率、移动滑动平局等地方需要使用glob\_step参数但并不会更改其值。


### LRN(Local Response Normalization)在caffe和tensorflow中实现上的差异

tensorflow的实现公式：
```math
b^{i}_{x,y}=a^{i}_{x,y}/(k+\alpha\sum_{j=max(o,i-r)}^{min(i+r,N-1)}(a^{j}_{x,y})^2)^{\beta}
```
caffe的实现公式：
```math
b^{i}_{x,y}=a^{i}_{x,y}/(k+\alpha/n\sum_{j=max(o,i-n/2)}^{min(i+n/2,N-1)}(a^{j}_{x,y})^2)^{\beta}
```

* a表示第i个核在位置（x,y）运用激活函数ReLU后的输出
* n(或2r+1)是同一位置上临近的kernal map的数目
* N是kernal的总数
* `$k,\alpha,\beta$`都是超参数


caffe中的local\_size必须是奇数，等于公式中的n，而tensorflow中的depth\_radius等于n/2，因此二者的关系为：local\_size = 2*depth\_radius + 1。

alpha的定义caffe要在实际的系数上乘以local\_size