#

## lecture 2

### 一些基本运算

#### 例子： How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s

total_flops = 6 * 70e9 * 15e12
h100_flop_per_second = 1979e12 / 2
mfu = 0.5
flops_per_day = h100_flop_per_second * 1024 * 3600 * 24 * mfu
total_days = total_flops / flops_per_day

#### 线性模型

假设你有一个线性模型：
- n 个点
- 每个点有 d 个特征
- 模型对每一个 d 维向量都有一个 k 维输出

$flops = n * d * k$

更近一步：
- n就是token数
- d*k就是模型参数数目
- 最终前向传播的flops数就是 2 (#token)(#paramenters)
- 反向传播的flops数也是 4 (#token)(#paramenters)

#### 计算强度

#### 加速强度