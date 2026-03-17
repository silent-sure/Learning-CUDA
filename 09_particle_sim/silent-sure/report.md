# 粒子物理模拟与可视化 总结报告

江彦韬（ID：silent-sure）

## 一、实现思路

### 1. 物理模型

每个粒子在洛伦兹力与来自其他粒子的库仑力的作用下运动，此运动可用下面的方程描述：
$$
\frac{\text d\mathbf r_i}{\text dt}=\mathbf v_i \\
m_i\frac{\text d\mathbf v_i}{\text dt}=q_i(\mathbf E_i + \mathbf v_i\times\mathbf B) \\
\mathbf E_i=k\sum_{j\ne i}\frac{q_j(\mathbf r_i-\mathbf r_j)}{|\mathbf r_i-\mathbf r_j|^3}
$$

### 2. 数值方法

Boris 算法 [1, 2] 是处理带电粒子在磁场与电场的作用下运动的重要方法，以简洁、准确、稳定著称。

Boris 算法的核心步骤是在每一步以下方式推进一个粒子：

- 前半步电场作用
$$\mathbf v^- \leftarrow \mathbf v^{(n-1/2)}+\frac{q\Delta t}{2m}\mathbf E^{(n)}$$
- 磁场引发的旋转
$$
\mathbf t\leftarrow\frac{q\Delta t}{2m}\mathbf B^{(n)}\\
\mathbf v'\leftarrow\mathbf v^- + \mathbf v^-\times \mathbf t\\
\mathbf v^+\leftarrow\mathbf v^-+\frac2{1+\mathbf t^2}(\mathbf v'\times\mathbf t)
$$
- 后半步电场作用
$$
\mathbf v^{(n+1/2)}\leftarrow \mathbf v^++\frac{q\Delta t}{2m}\mathbf E^{(n)}
$$
- 更新位置
$$
\mathbf r^{(n+1)}\leftarrow \mathbf r^{(n)} + \Delta t\,\mathbf v^{(n+1/2)}
$$

Boris 算法通常处理的是 $\mathbf E^{(n)}$ 只与 $\mathbf r^{(n)}$ 有关而与其他粒子的位置无关的情形；在这种情形下，该算法是二阶准确的 [3]。然而，在此问题中，$\mathbf E^{(n)}$ 与其他粒子的位置有关（见“1. 物理模型”中的公式），而其他粒子的位置都在变化，这可能导致 Boris 算法的精度下降。不过，由于粒子的位置在一个时间步长内变化不大，我们仍然认为在此问题下使用 Boris 算法是比较准确的。

### 3. 存储与 CUDA 并行计算

用多数组结构（Structure of Arrays, SoA）存储粒子的质量 $m$、电荷量 $q$、位置 $\mathbf r$、速度 $\mathbf v$，其中标量用 ``float`` 存储，矢量用 ``float3`` 存储。

每个线程来模拟一个粒子的运动。每过一个时间步长，都要调用一次 ``cudaDeviceSynchronize()`` 来同步所有线程，保证下一步读取到的是正确的信息。

### 4. 模拟结果的记录

每 ``record_interval`` 个时间步长记录一次各个粒子的位置。每次记录时，将粒子的位置从设备端复制到主机端，并输出到文本文件或二进制文件中（由 TEXT_FORMAT 开关控制）。

## 二、优化方法

### 1. 强制内联

所有矢量计算函数与单个粒子的 Boris 推进函数都用 ``__forceinline__`` 强制内联，减少函数调用开销。

### 2. 只将位置传回

传回粒子的状态时，只需要传回位置 $\mathbf r$。

## 三、开发中发现的问题

### 1. 竞争条件（race condition）

``updateParticlesKernel`` 同时读和写文件；一般情况下，这时需要考虑竞争条件问题。然而，用当前时刻还是下一时刻的粒子位置计算库仑力对结果的影响不大，且所有粒子的信息都在每个时间步长得到同步，因此不需要特别处理。

### 2. 除以零

为了避免除以零，在两个粒子距离特别近（小于 ``1e-4``）时，视其距离为 ``1e-4`` 计算库仑力。

## 四、正确性验证

用以下输入文件测试单个粒子在磁场中的运动:

particles
```
0 -4 0 -2 0 0.159154943 1e-6 1e-6
```
mag

```
0 0 1
```

测得回旋半径为 $2 \text m$，螺距为 $1 \text m$，周期为 $2\pi\text s$，符合理论值。

## 五、性能指标与分析

模拟 10000 个时间步长，用时 126 s，平均每秒能模拟 79.4 个步长。

## 参考文献


[1] Boris, J. P. (1970). Relativistic Plasma Simulation-Optimization of a Hybrid Code. *Proc. Fourth Conference on the Numerical Simulation of Plasmas*, 3–67.  
[2] Birdsall, C. K. & Langdon, A. B. (1991). *Plasma Physics Via Computer Simulation*, pp. 57-61. Adam Hilger, UK.  
[3] Qin, H., Zhang, S., Xiao, J., Liu, J., Sun, Y., Tang, W. M. (2013). Why is Boris algorithm so good?. *Phys. Plasmas* **20** (8): 084503. doi:[10.1063/1.4818428](https://pubs.aip.org/aip/pop/article/20/8/084503/317652/Why-is-Boris-algorithm-so-good).