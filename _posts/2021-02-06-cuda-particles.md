---
title:  "CUDA实现简易粒子模拟"
date:   2021-02-06 18:24 +0100
categories: cuda
---

最近在上学校GPU Programming的课，这次作业是写一个简单的粒子系统，在此简单记录一下。

## 1. Verlet积分

$ \textbf{p}_i^{(t+1)} = 2 \textbf{p}_i^{(t)} - \textbf{p}_i^{(t-1)} + \Delta_t^2 a_i $

$\textbf{p}$为粒子的坐标。$a$为加速度。

## 2. 粒子与包围盒碰撞

当粒子的位置超出了包围盒的区域，则认为粒子与包围盒发生了碰撞。下图左上方的圆为粒子当前位置，右下角为根据Verlet积分得到的下一时刻的位置，那么在无能量损失的情况下，粒子下一时刻的实际位置为右上角的圆。

一种简易的考虑能量损失的方法是将反弹后的高度乘上一个小于1的系数$\lambda$，如下图所示：

如果在这一个时间步内，粒子将和包围盒多次发生碰撞（用上面的方法得到的粒子新位置仍然在包围盒外），那么首先需要判断发生碰撞的顺序。如下图，粒子先与垂直的平面碰撞，再与水平平面碰撞。

### 代码实现

这里假设包围盒是六面都与坐标轴垂直的长方体，x0,y0,z0是包围盒坐标的最小值，x1,y1,z1是最大值。r为粒子半径，bounce为上述的$\lambda$。我在这里使用直线的参数方程来判断粒子与哪一个平面先发生碰撞。参数为0是粒子的当前位置，为1是粒子下一时刻的位置，那么求出直线与六个平面的交点的参数t，大于0小于1的最小的t则是最先发生碰撞的位置。

```cpp
// collision with the bbox
float min_t; int min_idx;
float t[6];
while (next.x < x0 + r || next.y < y0 + r || next.z < z0 + r || next.x > x1 - r || next.y > y1 - r || next.z > z1 - r) { // if outside the bounding box
	// use parametirc equation of the line to determine the intersection order 
	// min_idx: the first plane the line intersects
	min_idx = -1;
	min_t = 2;
	t[0] = (next.x == cur_x || next.x >= x0 + r) ? 2 : (x0 + r - cur_x) / (next.x - cur_x);
	t[1] = (next.y == cur_y || next.y >= y0 + r) ? 2 : (y0 + r - cur_y) / (next.y - cur_y);
	t[2] = (next.z == cur_z || next.z >= z0 + r) ? 2 : (z0 + r - cur_z) / (next.z - cur_z);
	t[3] = (next.x == cur_x || next.x <= x1 - r) ? 2 : (x1 - r - cur_x) / (next.x - cur_x);
	t[4] = (next.y == cur_y || next.y <= y1 - r) ? 2 : (y1 - r - cur_y) / (next.y - cur_y);
	t[5] = (next.z == cur_z || next.z <= z1 - r) ? 2 : (z1 - r - cur_z) / (next.z - cur_z);
	for (int i = 0; i < 6; ++i) {
		if (t[i] >= 0 && t[i] <= 1 && t[i] < min_t) {
			min_idx = i;
			min_t = t[i];
		}
	}
	assert(min_idx != -1);

	switch (min_idx) {
	case 0:
		next.x = x0 + r + bounce * (x0 + r - next.x);
		cur_x = x0 + r - bounce * (cur_x - x0 - r);
		break;
	case 1:
		next.y = y0 + r + bounce * (y0 + r - next.y);
		cur_y = y0 + r - bounce * (cur_y - y0 - r);
		break;
	case 2:
		next.z = z0 + r + bounce * (z0 + r - next.z);
		cur_z = z0 + r - bounce * (cur_z - z0 - r);
		break;
	case 3:
		next.x = x1 - r - bounce * (next.x - x1 + r);
		cur_x = x1 - r + bounce * (x1 - r - cur_x);
		break;
	case 4:
		next.y = y1 - r - bounce * (next.y - y1 + r);
		cur_y = y1 - r + bounce * (y1 - r - cur_y);
		break;
	case 5:
		next.z = z1 - r - bounce * (next.z - z1 + r);
		cur_z = z1 - r + bounce * (z1 - r - cur_z);
		break;
	}
}
```

## 3. 粒子间碰撞

如果当前两个粒子间的距离小于它们的半径之和，则认为发生了碰撞（这是一种简化，实际上由于时间步是离散的，粒子的碰撞有可能发生在两个time step之间）。为了检测有无碰撞，需要遍历所有的两两粒子对，时间复杂度为$O(N^2)$。为了加速，通常使用空间划分--将空间划分为立方体格，将每个粒子归在其中心位置所在的格子（cell）内。这样对于每个粒子，需要检测其临近的cell内的粒子有无与其发生碰撞。 当cell的尺寸大于最大的粒子的直径时，就只用检测以当前cell为中心的3x3x3的立方体一共27个cell。

还有一种方法是将粒子归在其所有覆盖到的cell，这样对于每个粒子就只需检测其中心所在的cell内的所有其他粒子（比如上图中，第4,5,8,9的cell都需要将3号粒子包括），根据[1]，这样做效率会略微降低，所以还是采用上面那种方法。

那么怎样才能给定一个cell，快速找到位于这个cell内部的所有粒子呢？在C++里，也许首先会想到对每个cell建一个vector或者其他容器。但是在CUDA里没有这样的可变长度的数据结构，一个解决方法是设定每个cell内粒子数的上限，这样就能为每个cell分配定长的空间了。一个更高效的做法是建立一个prefix sum。即，将粒子的id根据其所在cell的id进行排序，排序后，在同一cell内的粒子在存储空间上是连续的，再保存每个cell的开始与结束位置，就能很快地找到任意cell内所有的粒子了。

### 代码实现

这里只挑选我认为比较关键的代码片段。完整代码可见我的github或者CUDA官方的sample。

首先对每个粒子，根据其空间位置计算其所属的cell_id。这里同时将粒子的id写入了particle_id，这是为了之后的排序。

``` cpp
__global__
void calc_cell_id(float* position, uint* cell_id, uint* particle_id, std::size_t num_particles) {
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_particles) return;
	float x, y, z;
	x = position[tid * 4 + 0];
	y = position[tid * 4 + 1];
	z = position[tid * 4 + 2];
	int xx, yy, zz;
	xx = static_cast<int> ((x - bb_min[0]) / dev_grid.x_size);
	yy = static_cast<int> ((y - bb_min[1]) / dev_grid.y_size);
	zz = static_cast<int> ((z - bb_min[2]) / dev_grid.z_size);

	cell_id[tid] = zz* dev_grid.x_num * dev_grid.y_num + yy * dev_grid.x_num + xx;
	particle_id[tid] = tid;
}
```
使用thrust库来对cell_id排序。注意这里是以cell_id为key，对cell_id和particle_id同时排序。
``` cpp
thrust::sort_by_key(thrust::device_ptr<uint>(cell_id), thrust::device_ptr<uint>(cell_id + num_particles), thrust::device_ptr<uint>(particle_id));
```
接下来要找到每个cell的开始和结束位置，分别存在start和end中。使用GPU就不用遍历了，只需要每个thread去判断当前位置和前一位置是否属于同一cell，如果不是，则说明当前位置是cell的开始位置。这里可以使用shared memory进行加速。因为每一个thread要读取相邻两个位置的值，那么相邻的thread就会重复读取一个位置，所以先读入shared_memory可以省去从global_memory中读两次的高耗时。
```cpp
__global__
void find_start(uint* start, uint* end, uint* cell_id, uint* particle_id, std::size_t num_particles) {
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_particles) return;
	__shared__ int shared_cell_id[1025];

	shared_cell_id[threadIdx.x+1] = cell_id[tid];
	if (threadIdx.x == 0) {
		if (tid != 0)
			shared_cell_id[0] = cell_id[tid - 1];
		else
			shared_cell_id[0] = 0;
	}
	__syncthreads();
	uint prev = shared_cell_id[threadIdx.x], cur = shared_cell_id[threadIdx.x + 1];
	if (prev != cur) {
		start[cur] = tid;
		end[prev] = tid;
	}
	if (tid == num_particles - 1)
		end[cur] = num_particles;
}
```
然后就可以对每个粒子检测其临近cell内的粒子是否与其碰撞了。
```cpp
__global__ void update_particles_kernel(float* position, float* cur_pos, float* last_pos,
	uint* cell_id, uint* particle_id, uint* start, uint* end, float dt, std::size_t num_particles) {
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_particles) return;

	// unwrap parameters
	float x0 = dev_params.bb_min[0], y0 = dev_params.bb_min[1], z0 = dev_params.bb_min[2];
	float x1 = dev_params.bb_max[0], y1 = dev_params.bb_max[1], z1 = dev_params.bb_max[2];
	float bounce = dev_params.bounce;

	// ... //

	cid = cell_id[tid];
	int3 cid_xyz = get_cell_index(cid);
	for (int cx = max(cid_xyz.x - 1, 0); cx <= min(cid_xyz.x + 2, dev_grid.x_num); cx++)
	for (int cy = max(cid_xyz.y - 1, 0); cy <= min(cid_xyz.y + 2, dev_grid.y_num); cy++)
	for (int cz = max(cid_xyz.z - 1, 0); cz <= min(cid_xyz.z + 2, dev_grid.z_num); cz++) {

		cid2 = cz * dev_grid.x_num * dev_grid.y_num + cy * dev_grid.x_num + cx;
		for (int i = start[cid2]; i < end[cid2]; i++) {
			if (i == tid)
				continue;
			pid2 = particle_id[i];
			cur2 = make_float3(cur_pos[pid2], cur_pos[pid2 + num_particles], cur_pos[pid2 + 2 * num_particles]);
			r2 = cur_pos[pid2 + 3 * num_particles];
			p_ab = cur2 - cur;
			p_ab_norm = sqrt(dot(p_ab, p_ab)); // 两个粒子的距离
			if (p_ab_norm < r + r2) {
				// 相互作用力的计算
			}
		}
	}
// 计算粒子的新位置等 
}
```

## 4. 可能的bug

第一遍实现的时候，模拟的结果肉眼观察上没有问题，但是与老师的reference output差别不小。仔细检查，测试了每一个功能模块都没有找到问题。最后发现是update_kernel在读写global memory时产生的bug。

由于计算粒子相互碰撞的力需要用到两个粒子当前的位置以及上一帧的位置（用来计算速度）。在我最早的实现中，update_kernel的最后将粒子的新位置直接写入了global memory。可能存在的问题是，之后的thread在计算粒子碰撞时，读到的粒子位置可能是已经更新了的。正确的实现方法应该是创建两片global memory，在kernel中读和写使用不同的memory。

在CUDA中，一定要注意内存的读写，要想起不同的thread开始执行时间不同。要检查thread之间是否独立。


## reference

\[1\] [http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf](http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf)