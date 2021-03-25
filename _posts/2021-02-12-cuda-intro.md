---

title:  "CUDA programming model--CUDA的一些基本概念"
date:   2021-02-12 20:59 +0100
categories: cuda
---

在CUDA中，常常能听到thread, block, warp, grid, core, kernel, SM等名词。这些概念很容易产生混淆，本文尽量解释清它们之间的关系与区别。  

## 1. 软件视角
如果我们不去管GPU的硬件，只关注怎么写代码，怎么用CUDA runtime API。最基本的概念就是kernel，也就是以```__global__ void ```开头的一个像C++的函数的东西。由于kernel是多个GPU线程并行，在host端代码调用kernel时，要指定线程数，例如```compute<<<32, 256>>(args); ```。32是block数（num_block），256是每个block的thread数（thread_per_block）。block可以认为是许多thread组成的更高层级的一个单位。总的线程数则是thread_per_block × num_block。grid则是比block更高一级的单位，一个grid由一个kernel内所有的block组成，如下图所示。

总结下来。kernel是类似于函数的一个CUDA基本执行单位。block和grid则是两个CUDA Runtime API的概念，来对thread进行分组，在实际的硬件中并没有所谓的block和grid。

![grid-of-thread-blocks](https://i.ibb.co/jb8p8Sg/grid-of-thread-blocks.png)

## 2. 硬件视角

SM（streaming multiprocessor）在GPU硬件中用于创建，管理，调度thread。每个GPU由多个SM组成，比如Titan V有80个SM。在硬件中，SM不直接对单个thread操作，而是对由32个thread组成的warp进行操作。SM对warp的操作类似于SIMD，也就是一个warp中的thread同时执行同样的指令，如果有分支，thread要执行不同的指令，那么它们将依次执行，一个thread在执行时，其它thread被disable。

（从Volta架构之后，有了Independent Thread Scheduling，每个thread有自己的program counter和call stack，能在不牺牲太多效率的情况下让thread进行更灵活的操作，比如说分支。不像之前的架构，每个warp共享一个program counter。）

warp的执行所需要的上下文（program counter，register等）都被写在芯片上。因此SM可以无代价地切换warp来执行。warp由warp scheduler调度。如下图，warp0由于要访问内存（速度很慢），因此将其挂起，让warp1执行。warp1执行完之后（或者也被挂起后），warp2或warp3就会开始执行。根据GPU架构的不同，一个SM内可能有一个或多个warp scheduler。

![warp-scheduler](https://i.ibb.co/M8JP1d3/20210212195249.jpg)

SM中最多能有的warp（thread）数量是有限的。比如从CC（compute capability）3.5-CC7.2，每个SM最多有64个warp（2048个thread）。同时，所有warp要求的shared memory，register数量不能超过SM有的shared memory，register的数量。

GPU的core数这一名词一般只在商业上使用。实际上是指最大warp数。比如Titan V有80个SM，每个SM最多有64个warp，那core数（warp数）就是80*64=5120。

## 3. Block的调度

在硬件层面时，block会被划分为warp。在一个block中，从0号thread开始，每个warp分割连续index的32个thread。

每个block只会由同一个SM进行处理（也就意味着一个SM的shared memory，register数量至少要够一个block用），一个SM可以处理多个block。同一个SM上，多个block由一个queue组织，顺序进行处理。

![gpu-structure](https://i.ibb.co/xXz69Vd/20210212204242.jpg)

这意味着，只有在一个block中，thread之间才能进行通信（shared memory）与同步。不同block需要保持独立。

## 4. num_block和thread_per_block的选择

num_block也就是grid_dim。thread_per_block就是block_dim。

在设置kernel的num_block和thread_per_block时需要考虑硬件。比如说要执行66个thread（假设只有一个SM），我们可以用2个block，每个block33个thread，也可以用3个block，每个block32个thread。乍一看可能觉得后者效率低，因为有只需要执行66个thread，kernel确调用了96个thread。但由于warp（32个thread组成）才是基本的执行单位，第一种方法里，每个block有33个thread，那就需要2个warp，2个block则一共4个warp。而第二种方法只需要3个warp。所以应该采用第二种。

除了要考虑warp，还要考虑SM的总数。比如我们有8个SM，任务需要10240个thread。可以用10个block，每个block有1024个thread。那首先8个SM各处理完一个block，之后由2个SM处理剩下的2个block。我们也可以用16个block，每个block 640thread。这样8个SM各处理两个block。这两种方法的耗时都是SM处理两个block的耗时，但第二种的block size更小，所以总耗时更短。

## Reference

\[1\][http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
\[2\][https://medium.com/@shiyan/some-cuda-concepts-explained-12ecc390d10f](https://medium.com/@shiyan/some-cuda-concepts-explained-12ecc390d10f)
