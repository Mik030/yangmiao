# YOLO v1～v5

## YOLO v1

论文：https://arxiv.org/abs/1506.02640

代码：https://github.com/pjreddie/darknet

**创新点**：

* 将整张图片作为网络的输入，直接在输出层对BBox的位置和类别进行回归。

![image-20201112151610087](https://tva1.sinaimg.cn/large/0081Kckwly1gkmevaguxnj3110080n9g.jpg)

### 简介

YOLO意思是You Only Look Once，创造性的将候选区和对象识别这两个阶段合二为一，属于one-stage的检测模型。

整体上来说，首先将图片resize到448×448，送入到CNN网络之后，经过进一步预测得到检测的结果。YOLO是用了一个单独的CNN模型来实现了end2end的检测，它是一个统一的框架，而且检测是实时的迅速的，所以论文的全称就是：You Only Look Once: Unified, Real-Time Object Detection。

### 设计理念

实际上，YOLO并没有真正去掉候选区，而是采用了预定义的候选区。YOLO首先将resize之后的图片分为了**$S\times S$个grid cell**，每个小网格的任务就是去检测那些——中心落在其中的object。

<img src="https://tva1.sinaimg.cn/large/0081Kckwly1gkmfbgu0prj312u0ocx65.jpg" alt="image-20201112153141822" style="zoom:40%;" />

具体地可以参照上图来理解，比如图中白色轿车的中心落在了第二行倒数第二列的单元格中，那么这个单元格就负责来预测这辆车。

对于所有的小单元格，它会预测$B$个**bbox**，和**bbox的置信度**（confidence score），输出为$(x,y,w,h,c)$。

其中 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 是边界框的中心坐标，而 ![[公式]](https://www.zhihu.com/equation?tex=w) 和 ![[公式]](https://www.zhihu.com/equation?tex=h) 是边界框的宽与高。还有一点要注意，中心坐标的预测值 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 是相对于每个单元格左上角坐标点的偏移值，并且单位是相对于单元格大小的，而边界框的 ![[公式]](https://www.zhihu.com/equation?tex=w) 和 ![[公式]](https://www.zhihu.com/equation?tex=h) 预测值是相对于整个图片的宽与高的比例，这样理论上4个元素的大小应该在 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 范围。

置信度的含义在MAL笔记中解释过，它是每个bbox输出的一个重要参数，它的定义有两重：

* bbox内有对象的概率，注意，是对象，不是某个类别的对象，也就是说只能分别background和object。如果这个bbox是背景，那么 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D0)。而当该边界框包含目标时， ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D1)。
* 网络预测的bbox与真实区域的IoU

用公式来表示置信度就是$ \begin{aligned} C_{i}^{j} & = P_{r}(Object) * IoU_{pred}^{truth} \ \end{aligned} $。综合来说，一个bounding box的置信度Confidence意味着它是否包含对象且位置准确的程度。置信度高表示这里存在一个对象且位置比较准确，置信度低表示可能没有对象或者即便有对象也存在较大的位置偏差。

除此之外，每个单元格还要给出预测的结果，输出的是$C$个类别的概率值，这些概率值其实是在各个边界框置信度下的条件概率，即 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28class_%7Bi%7D%7Cobject%29)。在这里，YOLO的一个缺点是，不管一个单元格预测多少个边界框，其只预测一组类别概率值。在后来的改进版本中，Yolo9000是把类别概率预测值与边界框是绑定在一起的。

由此，在bbox内有object的条件下，我们可以计算出类别的置信度![[公式]](https://www.zhihu.com/equation?tex=Pr%28class_%7Bi%7D%7Cobject%29%2APr%28object%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D%3DPr%28class_%7Bi%7D%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D)。

边界框类别置信度，表征的是该边界框中目标属于各个类别的可能性大小，以及边界框匹配目标的好坏。之后设置阈值，过滤掉得分低的bbox，对剩下的部分进行NMS，得到最终的检测结果。

总结一下，每个单元格需要预测 ![[公式]](https://www.zhihu.com/equation?tex=%28B%2A5%2BC%29) 个值。如果将输入图片划分为 ![[公式]](https://www.zhihu.com/equation?tex=S%5Ctimes+S) 网格，那么最终预测值为 ![[公式]](https://www.zhihu.com/equation?tex=S%5Ctimes+S%5Ctimes+%28B%2A5%2BC%29) 大小的张量。整个模型的预测值结构如下图所示。对于PASCAL VOC数据，其共有20个类别，如果使用 ![[公式]](https://www.zhihu.com/equation?tex=S%3D7%2CB%3D2) ，那么最终的预测结果就是 ![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes+7%5Ctimes+30) 大小的张量。

### 网络结构

![image-20201113101417256](https://tva1.sinaimg.cn/large/0081Kckwly1gknbrhxjeij31gy0mg42m.jpg)

YOLO使用卷积层来提取特征，然后使用全连接层来得到预测的结果。具体在PASCAL VOC上来看，主要是使用了1x1卷积来做channle reduction，然后紧跟3x3卷积，做max pooling。对于卷积层和全连接层，采用Leaky ReLU激活函数，但是最后一层采用线性激活函数。最终输出一个7×7×30的tensor。

<img src="https://tva1.sinaimg.cn/large/0081Kckwly1gkncah58puj30il06dq49.jpg" alt="image-20201113103231699" style="zoom:70%;" />

来看看这个30维的向量都包含哪些信息。

![image-20201113103459388](https://tva1.sinaimg.cn/large/0081Kckwly1gkncmehw7zj30ji0b8q3j.jpg)

具体来看每个部分：

* **20个对象分类的概率**

因为YOLO支持识别20种不同的对象（人、鸟、猫、汽车、椅子等），所以这里有20个值表示该网格位置存在任一种对象的概率。可以记为 ![[公式]](https://www.zhihu.com/equation?tex=P(C_1|Object)%2C+......%2C+P(C_i|Object)%2C......P(C_{20}|Object)) ，之所以写成条件概率，意思是如果该网格存在一个对象Object，那么它是 ![[公式]](https://www.zhihu.com/equation?tex=C_i) 的概率是 ![[公式]](https://www.zhihu.com/equation?tex=P(C_i|Object)) 。

* **2个bounding box的位置**

也就是![[公式]](https://www.zhihu.com/equation?tex=(x%2C+y%2Cw%2Ch))的编码。对于边界框为什么把置信度 ![[公式]](https://www.zhihu.com/equation?tex=c) 和 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C+y%2Cw%2Ch%29) 都分开排列，而不是按照 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C+y%2Cw%2Ch%2Cc%29) 这样排列，其实纯粹是为了计算方便，因为实际上这30个元素都是对应一个单元格，其排列是可以任意的。但是分离排布，可以方便地提取每一个部分。

首先网络的预测值是一个二维张量 ![[公式]](https://www.zhihu.com/equation?tex=P) ，其shape为 ![[公式]](https://www.zhihu.com/equation?tex=%5Bbatch%2C+7%5Ctimes+7%5Ctimes+30%5D) 。采用切片，那么 ![[公式]](https://www.zhihu.com/equation?tex=P_%7B%5B%3A%2C0%3A7%2A7%2A20%5D%7D) 就是类别概率部分，而 ![[公式]](https://www.zhihu.com/equation?tex=P_%7B%5B%3A%2C7%2A7%2A20%3A7%2A7%2A%2820%2B2%29%5D%7D) 是置信度部分，最后剩余部分 ![[公式]](https://www.zhihu.com/equation?tex=P_%7B%5B%3A%2C7%2A7%2A%2820%2B2%29%3A%5D%7D) 是边界框的预测结果。

* **2个bounding box的置信度**

 bounding box的置信度![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+Confidence+%3D+Pr%28Object%29++%2A+IOU_%7Bpred%7D%5E%7Btruth%7D+)

![[公式]](https://www.zhihu.com/equation?tex=Pr%28Object%29) 是bounding box内存在对象的概率，区别于上面第①点的 ![[公式]](https://www.zhihu.com/equation?tex=P%28C_i%7CObject%29)。  ![[公式]](https://www.zhihu.com/equation?tex=Pr%28Object%29)并不管是哪个对象，它体现的是有或没有对象的概率。第①点中的 ![[公式]](https://www.zhihu.com/equation?tex=P%28C_i%7CObject%29) 意思是假设已经有一个对象在网格中了，这个对象具体是哪一个。

还要说明的是，虽然有时说"预测"的bounding box，但这个IOU是在训练阶段计算的。等到了测试阶段（Inference），这时并不知道真实对象在哪里，只能完全依赖于网络的输出，这时已经不需要（也无法）计算IOU了。

每个30维向量中只有一组（20个）对象分类的概率，也就只能预测出一个对象。所以输出的 7*7=49个 30维向量，**最多表示出49个对象**。

### 网络训练

预训练的的分类模型，采用了GoogleNet的前20层，然后添加一个average-pool层和全连接层。

预训练之后，在预训练得到的20层卷积层之上加上随机初始化的4个卷积层和2个全连接层。

![image-20201113111632735](https://tva1.sinaimg.cn/large/0081Kckwly1gkndkago9wj31240lcdss.jpg)

### 损失函数

在实现中，最主要的就是怎么设计损失函数，让这个三个方面得到很好的平衡。Yolo算法将目标检测看成回归问题，所以采用的是均方差损失函数。但是对不同的部分采用了不同的权重值。作者简单粗暴的全部采用了**sum-squared error loss**来做这件事。

![image-20201116084349551](https://tva1.sinaimg.cn/large/0081Kckwly1gkqq0bvi4aj30j00bw0y4.jpg)

第一项是边界框中心坐标的误差预测， ![[公式]](https://www.zhihu.com/equation?tex=1%5E%7Bobj%7D_%7Bij%7D) 表示第![[公式]](https://www.zhihu.com/equation?tex=i)个单元格是否存在目标，该单元格中的第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个边界框负责预测该目标。

第二项是边界框的高与宽的误差项，并采用了较大的权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda+_%7Bcoord%7D%3D5)。宽度和高度先取了平方根，因为如果直接取差值的话，大的对象对差值的敏感度较低，小的对象对差值的敏感度较高，所以取平方根可以降低这种敏感度的差异，使得较大的对象和较小的对象在尺寸误差上有相似的权重。

第三项是包含目标的边界框的置信度误差，![[公式]](https://www.zhihu.com/equation?tex=1%5E%7Bobj%7D_%7Bi%7D) 指的是第![[公式]](https://www.zhihu.com/equation?tex=i)个单元格是否存在目标。对于置信度 ![[公式]](https://www.zhihu.com/equation?tex=C_i) ，如果是不存在目标，此时由于 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D0)，那么 ![[公式]](https://www.zhihu.com/equation?tex=C_i%3D0) 。如果存在目标， ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D1) ，此时需要确定 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D) ，当然你希望最好的话，可以将IOU取1，这样 ![[公式]](https://www.zhihu.com/equation?tex=C_i%3D1)。实际的复现中，也是经常直接取为1，这个带来的影响应该不是很大。

第四项是不包含目标的边界框的置信度误差，采用较小的权重值 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda+_%7Bnoobj%7D%3D0.5)。

第五项是对类别的预测，但前提也是框中有目标。

所以，**对于不存在对应目标的边界框，其误差项就是只有置信度**，坐标项误差是没法计算的。而**只有当一个单元格内确实存在目标时，才计算分类误差项**，否则该项也是无法计算的。

同时，在训练时，如果该单元格内确实存在目标，那么只选择与ground truth的IOU最大的那个边界框来负责预测该目标，而其它边界框认为不存在目标。这样设置的一个结果将会使一个单元格对应的边界框更加专业化，但**如果一个单元格内存在多个目标，这时候Yolo算法就只能选择其中一个来训练**，这也是Yolo算法的缺点之一。

### 总结

**优点**：

* 快速，实时性强-pipline简单，训练与预测都是end2end
* 背景误检率低-对整张图片做卷积，所以其在检测目标有更大的视野
* 通用性强

**缺点**：

* 对于**小物体的检测较差**，尤其是一些聚集在一起的小对象。

  对边框的预测准确度不是很高，总体预测精度略低于Fast RCNN。主要是因为**网格设置比较稀疏**，而且每个网格只预测两个边框，另外Pooling层会丢失一些细节信息，对定位存在影响。 























> 参考：
>
> https://zhuanlan.zhihu.com/p/46691043
>
> https://zhuanlan.zhihu.com/p/136382095
>
> https://zhuanlan.zhihu.com/p/32525231

































































