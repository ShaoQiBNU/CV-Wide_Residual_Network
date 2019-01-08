Wide Residual Network详解
==========================

# 一. 背景

> 深度学习发展至今，通过增加模型深度来加强模型的表达能力已经成为行业共识。Resnet网络是眼下最为成功，应用最为广泛的一种深度学习模型。Residual block中identity mapping的引入，使得模型可以将深度恣意扩展到很深，它直接将原来数十层网络伸展到了数百乃至上千层。
>
> 不过深度Resnet网络也有潜在的隐忧。虽然说它的模型可通过不断增加深度来获得能力提升（即分类准备性能提高），但愈到后期，靠单纯增加模型层数换来的边际性能提升愈低。随着模型深度的加深，梯度反向传播时，并不能保证能够流经每一个残差模块（residual block）的weights，以至于它很难学到东西，因此在整个训练过程中，只有很少的几个残差模块能够学到有用的表达，而绝大多数的残差模块起到的作用并不大。
>
> 来自巴黎大学的作者试图对Residual learning进行一番透彻的再思考，他们对Residual block的组成，里面的层结构、层数和层宽等进行了广泛的实验，最终他们表明加宽Resnet有时候比加深它带来的边际性能提升要更多。
>
> 当然加宽Residual block意味着会带来训练参数平方级数目的增加，同时训练参数过多也会导致模型易陷入overfitting的陷阱。作者通过在加宽后的Residual block里面插入Dropout层有效地避免了overfitting问题。最后多个数据集上的实验表明他们提出的Wide Residual network (WRN)可取得比那些细高的original Residual network更高的分类精度，同时也拥有着更快的训练速度。

# 二. 网络结构

## (一) residual block设计

> 扩展 residual block 的能力主要有以下三种方式：

```
1. 加深卷积层，即增加更多的卷积层，深度增加

2. 加宽卷积层，即扩大filters，使得channels数量增加

3. 使用更大的kernel size，由于 3x3 的卷积kernel size已经被许多研究证明有效，所以论文继续采用 3x3，不对这一方式进行尝试
```

## (二) residual block里的conv结构设计

> 下面涉及到的参数定义如下：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=l" target="_blank"><img src="https://latex.codecogs.com/svg.latex?l" title="l" /></a>表示residual block里卷积层的个数；
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k" title="k" /></a>表示residual block里卷积filters加宽的factor；
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?N" title="N" /></a>表示每个residual block的重复次数；
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d" title="d" /></a>表示residual block的个数；
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=depth" target="_blank"><img src="https://latex.codecogs.com/svg.latex?depth" title="depth" /></a>表示网络的深度，计算方式为：<a href="https://www.codecogs.com/eqnedit.php?latex=depth&space;=&space;d*l*N&plus;4" target="_blank"><img src="https://latex.codecogs.com/svg.latex?depth&space;=&space;d*l*N&plus;4" title="depth = d*l*N+4" /></a>，其中4代表第一个卷积层、最后一个全局平均pool层、pool之后的flatten层和全连接输出层；

![image](https://github.com/ShaoQiBNU/Wide_Residual_Network/blob/master/images/1.png)

> 如图所示，该网络结构的相关参数对应如下：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}l=2&space;\\&space;d=3&space;\\&space;depth&space;=&space;d*l*N&plus;4&space;=&space;6N&plus;4&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}l=2&space;\\&space;d=3&space;\\&space;depth&space;=&space;d*l*N&plus;4&space;=&space;6N&plus;4&space;\end{matrix}\right." title="\left\{\begin{matrix}l=2 \\ d=3 \\ depth = d*l*N+4 = 6N+4 \end{matrix}\right." /></a>

### 1. conv结构

> 以B(M)表示residual block的结构，M表示block里所含conv层的kernel size list。比如B(3,1)表示Block里先后包含两个分别为 3x3 与 1x1 大小的Conv层，以下为实验中所考虑的几种block结构。

```
1. B(3,3) - original basic block

2. B(3,1,3) - with one extra 1×1 layer

3. B(1,3,1) - with the same dimensionality of all convolutions, straightened bottleneck

4. B(1,3) - the network has alternating 1×1 - 3×3 convolutions everywhere

5. B(3,1) - similar idea to the previous block

6. B(3,1,1) - Network-in-Network style block
```

> 下图为以上各个结构最终的分类结果比较（注意在实验时作者为保证训练所用参数相同，因此不同类型block构成的网络的深度会有不同），从图中可见，B(3,3)能取得最好的结果，这也证明了常用Residual block的有效性。B(3,1,3)与B(3,1)性能略差，但速度却更快。接下来的实验中，作者保持了使用B(3,3)这种Residual block结构。

### 2. conv层次

> <a href="https://www.codecogs.com/eqnedit.php?latex=l" target="_blank"><img src="https://latex.codecogs.com/svg.latex?l" title="l" /></a>的大小即residual block内conv层数目。通过保持整体训练所用参数不变，作者研究、分析了residual block内conv层数目不同所带来的性能结果差异，具体对比结果如图所示，从中我们能够看出residual block里面包含2个conv层可带来最优的分类结果性能。

![image](https://github.com/ShaoQiBNU/Wide_Residual_Network/blob/master/images/2.png)

### 3. conv宽度

> <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k" title="k" /></a>表示Residual block的宽度因子，<a href="https://www.codecogs.com/eqnedit.php?latex=k=1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k=1" title="k=1" /></a>为原始resnet网络中的宽度。通过增加<a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k" title="k" /></a>来加宽residual blocks，并保持整体可训练参数数目不变，作者进行了广泛的实验。结果表明加大Resnet的宽度可带来比加大其深度更大的边际性能提升，具体结果可见下图：

![image](https://github.com/ShaoQiBNU/Wide_Residual_Network/blob/master/images/3.png)

### 4. dropout引入

> 加宽Residual block会带来训练参数的激增，为了避免模型陷入过拟合，作者在Residual block中引入了dropout，dropout被放在了convolutional_block中的ReLu之后，下图显示了dropout带来的性能提升：

![image](https://github.com/ShaoQiBNU/Wide_Residual_Network/blob/master/images/4.png)

# 三. 实验结果对比

> 作者对比了wide residual network和resnet在ImageNet和COCO数据集上的结果，具体见论文。无论是计算效率还是分类精度，wide residual network均优于ResNet。

# 四. 代码

> 实现了论文里CIFAR10分类的结构——WRN28-10，dropout为0.3，代码如下：

```python
######################## load packages ###########################
import tensorflow as tf
import numpy as np
import cifar10

######################## download dataset ###########################
cifar10.maybe_download_and_extract()

######################## load train and test data ###########################
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

######################## network paramaters ###########################
epochs = 20
train_batch_size = 128
test_batch_size = 100

learning_rate = 0.001
display_step = 20

########## set net parameters ##########
image_size = 32
num_channels = 3

#### 10 kind ####
n_classes = 10

#### widening factor ####
k = 10

########## placeholder ##########
x = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels])
y = tf.placeholder(tf.float32, [None, n_classes])


######################## random generate data ###########################
def random_batch(images, labels):
    '''
    :param images: 输入影像集
    :param labels: 输入影像集的label
    :return: batch data
    '''

    num_images = len(images)

    ######## 随机设定待选图片的id ########
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)

    ######## 筛选data ########
    x_batch = images[idx, :, :]
    y_batch = labels[idx, :]

    return x_batch, y_batch


##################### build net model ##########################
######### identity_block #########
def identity_block(inputs, filters, kernel, strides):
    '''
    identity_block: 两层的恒等残差块，影像输入输出的height和width保持不变，channel发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 两层恒等残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2 = filters
    k1, k2 = kernel
    s1, s2 = strides

    ######## shortcut 第一种规则，影像输入输出的height和width保持不变，输入直接加到卷积结果上 ########
    inputs_shortcut = inputs

    ######## first identity block 第一层恒等残差块 ########
    #### BN ####
    layer1 = tf.layers.batch_normalization(inputs)

    #### relu ####
    layer1 = tf.nn.relu(layer1)

    #### conv ####
    layer1 = tf.layers.conv2d(layer1, filters=f1, kernel_size=k1, strides=s1, padding='SAME')

    #### BN ####
    layer1 = tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1 = tf.nn.relu(layer1)

    #### dropout ####
    layer1 = tf.nn.dropout(layer1, 0.3)

    ######## second identity block 第二层恒等残差块 ########
    #### conv ####
    layer2 = tf.layers.conv2d(layer1, filters=f2, kernel_size=k2, strides=s2, padding='SAME')

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out = tf.add(inputs_shortcut, layer2)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


######## convolutional_block #########
def convolutional_block(inputs, filters, kernel, strides):
    '''
    convolutional_block: 两层的卷积残差块，影像输入输出的height、width和channel均发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 两层的卷积残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2 = filters
    k1, k2 = kernel
    s1, s2 = strides

    ######## shortcut 第二种规则，影像输入输出height和width发生变化，需要对输入做调整 ########
    #### conv ####
    inputs_shortcut = tf.layers.conv2d(inputs, filters=f1, kernel_size=1, strides=s1, padding='SAME')

    #### BN ####
    inputs_shortcut = tf.layers.batch_normalization(inputs_shortcut)

    ######## first convolutional block 第一层卷积残差块 ########
    #### conv ####
    layer1 = tf.layers.conv2d(inputs, filters=f1, kernel_size=k1, strides=s1, padding='SAME')

    #### BN ####
    layer1 = tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1 = tf.nn.relu(layer1)

    #### dropout ####
    layer1 = tf.nn.dropout(layer1, 0.3)

    ######## second convolutional block 第二层卷积残差块 ########
    #### conv ####
    layer2 = tf.layers.conv2d(layer1, filters=f2, kernel_size=k2, strides=s2, padding='SAME')

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out = tf.add(inputs_shortcut, layer2)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


######## ResNet CIFAR network #########
def WideResnet_CIFAR(x, n_classes, k):

    ####### first conv ########
    #### conv ####
    conv1 = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='SAME')

    #### BN ####
    conv1 = tf.layers.batch_normalization(conv1)

    #### relu ####
    conv1 = tf.nn.relu(conv1)


    ####### second conv ########
    #### convolutional_block 1 ####
    conv2 = convolutional_block(conv1, filters=[16*k, 16*k], kernel=[3, 3], strides=[1, 1])

    #### identity_block 2 ####
    conv2 = identity_block(conv2, filters=[16*k, 16*k], kernel=[3, 3], strides=[1, 1])


    ####### third conv ########
    #### convolutional_block 1 ####
    conv3 = convolutional_block(conv2, filters=[32*k, 32*k], kernel=[3, 3], strides=[2, 1])

    #### identity_block 3 ####
    conv3 = identity_block(conv3, filters=[32*k, 32*k], kernel=[3, 3], strides=[1, 1])


    ####### fourth conv ########
    #### convolutional_block 1 ####
    conv4 = convolutional_block(conv3, filters=[64*k, 64*k], kernel=[3, 3], strides=[2, 1])

    #### identity_block 5 ####
    conv4 = identity_block(conv4, filters=[64*k, 64*k], kernel=[3, 3], strides=[1, 1])


    ####### 全局平均池化 ########
    pool = tf.nn.avg_pool(conv4, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='VALID')

    ####### flatten 影像展平 ########
    flatten = tf.reshape(pool, (-1, 1 * 1 * 64 * k))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)

    return out


########## define model, loss and optimizer ##########
#### model pred 影像判断结果 ####
pred = WideResnet_CIFAR(x, n_classes, k)

#### loss 损失计算 ####
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


##################### train and evaluate model ##########################
########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(len(images_train) // train_batch_size):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = random_batch(images_train, labels_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(len(images_test) // test_batch_size):
        batch_x, batch_y = random_batch(images_test, labels_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
```



# 五. 参考

https://github.com/szagoruyko/wide-residual-networks.



https://www.jianshu.com/p/7150963d0e93.




