---

title:  "Robust Motion In-between论文笔记"
date:   2021-04-19 20:00 +0100
categories: paper
---


# Robust Motion In-between论文笔记

制作游戏中的角色动作有两类方法：由动画师制作关键帧（keyframes）或者动作捕捉（mocap）。
这两类方法都非常耗时，因此可以利用机器学习的方法来减少人工工作量。
mocap的缺陷是原始捕捉数据含有许多噪音，仍然需要人工清理，"Robust solving of optical motion capture data by denoising"这篇文章研究了对mocap数据降噪。
本文则针对的是keyframe流程的加速。  

由关键帧生成中间帧最原始的方法是直接插值。显然，当两个关键帧间隔较长时，插值的结果会很差。
本文提出的模型在测试时最多能预测100帧中间帧（动作为25fps，因此长达4秒）。

## 1. 模型结构

如图所示，模型的输入是前后两个关键帧，以及之前的10帧连续帧。所以预测完本段中间帧后，最后10帧会作为下一段预测的输入。

![img](https://i.ibb.co/ZmKZLRn/motion-in-between-1.png)

模型的大致结构如下图所示，state encoder的输入是当前帧（根节点速度q，关节及根节点的四元数q，脚是否接触地面r）的concatenation。
target encoder的输入是下一关键帧的旋转。offset encoder的输入是当前帧与下一关键帧的旋转及根节点位置的差。三个encoder为结构相同的MLP。
用LSTM来进行序列预测，最后经过decoder预测（q,c,r）。
    

![img2](https://i.ibb.co/nLqv5W3/motion-in-between-2.png)  

## 2. Loss functions

Loss包含两类：Reconstruction和Adversarial。Reconstruction Loss包括(q,c,r,从四元数和骨骼得到的关节位置p)与Ground truth间的L1 loss。
Adversarial则是针对motion训练了两个discriminator，一个discriminator以相邻10帧作为输入，另一个以相邻两帧作为输入。
Discriminator以滑动窗口的方式对所有预测的帧进行分类。

![img3](https://i.ibb.co/SfL3217/motion-in-between-3.png)

## 3. Embedding modifiers
#### Time-to-arrival 
作者发现单纯使用LSTM对预测不同长度的motion效果不佳。因此使用了positional encoding，与原本encoder输出的embedding相加。
positional encoding公式如下，其中tta是指当前帧与最后一帧（下一关键帧）的间隔，d为embedding的维度，i为embedding当前的下标，basis为常数（此处是10000）。  

![img5](https://i.ibb.co/h229nQx/motion-in-between-5.png)

将positional encoding的向量z可视化的结果为下图左边。作者提出，当要预测的帧数很多时，使用原始的positional encodings会使泛化性变差。
因为训练时很可能没有或很少有这么长要预测的序列，测试时会出现没见过的z，所以对z进行截断。
如果帧数超过了训练时最多的帧数减5帧，则将多出来的帧的z都设置为常量，如下图右边所示。

![img4](https://i.ibb.co/3098C3S/motion-in-between-4.png)

## 4. Scheduled target noise  
至此，在输入固定时，模型的输出也是固定的。在实际场景中，会希望输出能有一些变化能够调节。
所以在offset encoder和target encoder的输出上加上了一个噪音向量。
此外，该噪音向量还乘上了一个因子，使靠前的帧噪音大，靠后的帧噪音小。




