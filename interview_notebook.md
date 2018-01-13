# 面试准备笔记
@(收集箱)[格林深瞳, 201801, 0111]
>本片文章记录了格林深瞳实习面试中可能会考的一些试题，格林深瞳公司的面试风格目前为电话面试，主要面试问题围绕机器学习基础知识与算法题考察展开。

**简历**：[https://github.com/MingChaoXu/yart/](https://github.com/MingChaoXu/yart/blob/master/20171226.html)

***

[TOC]

## 问题整理
### 机器学习理论知识
#### 1.bagging

#### 2.过拟合的解决方法
详情见博客[过拟合解决办法](http://blog.csdn.net/taoyanqi8932/article/details/71101699)
#### 3.Google net/resnet
详情见[GoolgleNet与ResNet](http://blog.csdn.net/qq_31531635/article/details/61914305)
####  3.loss选择
详情见博客[loss函数](http://blog.csdn.net/luojun2007/article/details/78136615)
####  4.交叉熵和sigmoid搭配的优势
用sigmoid函数作为神经元的激活函数时，最好使用交叉熵代价函数，以避免训练过程太慢
详情见博客[sigmoid与交叉熵搭配的优势](http://blog.csdn.net/u014313009/article/details/51043064)
###  5.kmeans原理
  K-means算法是聚类分析中使用最广泛的算法之一。它把n个对象根据他们的属性分为k个聚类以便使得所获得的聚类满足：同一聚类中的对象相似度较高；而不同聚类中的对象相似度较小。其聚类过程可以用下图表示：
  <div align=center><img width="500" height="350" src="http://img.blog.csdn.net/20131024140835468"/></div>
  如图所示，数据样本用圆点表示，每个簇的中心点用叉叉表示。(a)刚开始时是原始数据，杂乱无章，没有label，看起来都一样，都是绿色的。(b)假设数据集可以分为两类，令K=2，随机在坐标上选两个点，作为两个类的中心点。(c-f)演示了聚类的两种迭代。先划分，把每个数据样本划分到最近的中心点那一簇；划分完后，更新每个簇的中心，即把该簇的所有数据点的坐标加起来去平均值。这样不断进行”划分—更新—划分—更新”，直到每个簇的中心不在移动为止。

该算法过程比较简单，但有些东西我们还是需要关注一下，此处，我想说一下"求点中心的算法"

一般来说，求点群中心点的算法你可以很简的使用各个点的X/Y坐标的平均值。也可以用另三个求中心点的的公式：
1. **Minkowski Distance 公式** —— λ 可以随意取值，可以是负数，也可以是正数，或是无穷大Distance 公式 —— λ 可以随意取值，可以是负数，也可以是正数，或是无穷大
$$ d_{ij} = \sqrt[\lambda]{\sum_1^n\mid x_{ik} - x_{jk} \mid ^ \lambda}$$
2. **Euclidean Distance 公式** —— 也就是第一个公式 λ=2 的情况
$$ d_{ij} = \sqrt{\sum_1^n\mid x_{ik} - x_{jk} \mid ^ 2}$$
3. **CityBlock Distance 公式** —— 也就是第一个公式 λ=1 的情况
$$ d_{ij} = \sum_1^n\mid x_{ik} - x_{jk} \mid $$

```python
@kmeans伪代码
begin initialization n, c, u1, u2,..., uc
do classify n samples according to nearest ui
	re-compute ui
until no change in ui
return u1, u2,..., uc 
```
**Kmeans算法的缺陷**

- 聚类中心的个数K 需要事先给定，但在实际中这个 K 值的选定是非常难以估计的，很多时候，事先并不知道给定的数据集应该分成多少个类别才最合适
- Kmeans需要人为地确定初始聚类中心，不同的初始聚类中心可能导致完全不同的聚类结果。（可以使用Kmeans++算法来解决）
### 6.KNN原理
算法思路：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。算法思路：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。
如下图：
 <div align=center><img width="300" height="300" src="http://img.blog.csdn.net/20131024142609828"/></div>
 
**算法流程**
从上图中我们可以看到，图中的数据集是良好的数据，即都打好了label，一类是蓝色的正方形，一类是红色的三角形，那个绿色的圆形是我们待分类的数据。
如果K=3，那么离绿色点最近的有2个红色三角形和1个蓝色的正方形，这3个点投票，于是绿色的这个待分类点属于红色的三角形
如果K=5，那么离绿色点最近的有2个红色三角形和3个蓝色的正方形，这5个点投票，于是绿色的这个待分类点属于蓝色的正方形
**KNN和K-Means的区别**
| KNN | K-Means |
| :-----| :-------|
| 1.KNN是分类算法<br>2.监督学习<br>3.喂给它的数据集是带label的数据，已经是完全正确的数据 |1.K-Means是聚类算法<br>2.非监督学习<br>3.喂给它的数据集是无label的数据，是杂乱无章的，经过聚类后才变得有点顺序，先无序，后有序|
|没有明显的前期训练过程，属于memory-based learning|有明显的前期训练过程|
|K的含义：来了一个样本x，要给它分类，即求出它的y，就从数据集中，在x附近找离它最近的K个数据点，这K个数据点，类别c占的个数最多，就把x的label设为c|K的含义：K是人工固定好的数字，假设数据集合可以分为K个簇，由于是依靠人工定好，需要一点先验知识|
**相似点**
都包含这样的过程，给定一个点，在数据集中找离它最近的点。即二者都用到了NN(Nears Neighbor)算法，一般用KD树来实现NN。

###  7.哈希过程。时间复杂度与空间复杂度

### 8.dropout的pytorch实现
### 9.weight decay、momentum与bn的作用
1. weight decay（权值衰减）的使用既不是为了提高你所说的收敛精确度也不是为了提高收敛速度，其最终目的是防止过拟合。在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
2. momentum是梯度下降法中一种常用的加速技术。对于一般的SGD，其表达式为$x\leftarrow x-\alpha *dx$，$x$沿负梯度方向下降。而带momentum项的SGD则写生如下形式：
$$v=\beta *v-a*dx$$
$$x\leftarrow x+v$$
其中$\beta$ 即momentum系数，通俗的理解上面式子就是，如果上一次的momentum（即v）与这一次的负梯度方向是相同的，那这次下降的幅度就会加大，所以这样做能够达到加速收敛的过程。
3. normalization。如果我没有理解错的话，题主的意思应该是batch normalization吧。batch normalization的是指在神经网络中激活函数的前面，将wx+b按照特征进行normalization，这样做的好处有三点：
- 提高梯度在网络中的流动。Normalization能够使特征全部缩放到[0,1]，这样在反向传播时候的梯度都是在1左右，避免了梯度消失现象。
- 提升学习速率。归一化后的数据能够快速的达到收敛。
- 减少模型训练对初始化的依赖。
### 项目相关
#### 1.LSTM反向传播推导
详情见博客[LSTM理解](https://www.jianshu.com/p/9dc9f41f0b29)
#### 2.决策树与决策森林与xgboost
详情见博客[xgboost简析](http://blog.csdn.net/sb19931201/article/details/52557382)
#### 3.整体的网络结构参数
<div align=center><img width="600" height="600" src="http://img.blog.csdn.net/20171218105952448?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvUXVpbmN1bnRpYWw=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast"/></div>

#### 4.CTC loss
**pytorch实现**
```python
cost = criterion(preds, text, preds_size, length) / batch_size
```
### 算法题整理
#### 1. 红绿灯时长估计
一个人为了估计红绿灯口的红灯、绿灯时长，但是不想每次等候一次完整的红绿灯，所以他记录了100天内每天走到路口的等待时长，问：如何利用这个数据来较为准确的估计红灯和绿灯的时长。
**思路**
可以先估计红灯的时长，因为绿灯的时长可以通过统计时长为0的次数与总次数之比作为概率来估计，红灯时长的估计方法有一种如下
$$ \frac{T}2=t_1+t_2+...+t_n $$
其中n为红灯的次数
#### 2. 给定一组数列，找出所有这组数列中和为m的两个数
#### 3. 图像如何产生运动模糊
通过椭圆滤波器算子
#### 4. 由均匀分布生成正太分布随机数
根据中心极限定理，数列之和服从正太分布
#### 5. 等概率生成线段
等概率的生成一组线段，要求线段的左端点大于等于a，右端点小于等于b
**思路**
将左端点和线段长度映射到二位空间，画出可行域





