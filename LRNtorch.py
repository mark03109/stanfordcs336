import math
import torch.nn.functional as F
import timeit
from typing import Iterable
import torch
from torch import nn
from einops import einsum, reduce, rearrange



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def einops_multi():
    x = torch.zeros(4, 8, device=device)
    assert x.dtype == torch.float32 # Dfault type
    assert x.numel() == 4 * 8
    assert x.element_size() == 4

    x = torch.ones(2, 3, 4, device=device)
    y = torch.ones(2, 3, 4, device=device)

    z1 = x @ y.transpose(-2, -1) # tranpose 交换倒数第二个轴和倒数第一个轴
    # @ 乘表示把最后两个维度矩阵相乘，前面维度看作批次

    z2 = einsum(x,y,"batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

    print(z1,"\n")
    print(z2,"\n")

def einops_reduce():
    x = torch.ones(2,3,4)

    y1 = x.sum(dim=-1)
    y2 = reduce(x, "... hidden -> ...", "sum")

    print(y1,"\n")
    print(y2,"\n")

def einops_rearrange():
    x = torch.ones(3, 8)
    w = torch.ones(4, 4)

    print(x, "\n")
    x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads = 2)
    print(x, "\n")

# gradient 梯度
def graidients_flops():
    B = 1024 # Number of points
    D = 256 # Dimension

    x = torch.ones(B, D, device = device)
    w1 = torch.randn(D, D, device = device, requires_grad=True)# 参与梯度计算
    w2 = torch.randn(D, D, device = device, requires_grad=True)

    # Forward pass
    h1 = einsum(x, w1, "batch in, in out -> batch out")
    h2 = einsum(h1, w2, "batch in, in out -> batch out")
    loss = (h2.mean() - 0)**2

    # Backward pass
    h1.retain_grad()# 指示h1 tensor在反向传播计算梯度时将梯度保存到h1.grade中
    h2.retain_grad()
    loss.backward()
    print(h1.shape,"\n",h1.grad,"\n")
    print(h2.shape,"\n",h2.grad,"\n")

    num_forwrd_flops = (2 * B * D * D)*2 # h2 = h1 @ w2 的计算数量

    h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
    assert torch.allclose(h1.grad, h1_grad)

    w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")
    assert torch.allclose(w2.grad, w2_grad)

    num_backward_flops = (2*B*D*D) * 2 # h2 = h1 @ w2 反向传播的计算数量

class Block(nn.Module):
    """Simple block tha applies a linear transformation followed by a ReLU nonlinearity"""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim,dim) / math.sqrt(dim))

    def forward(self, x : torch.Tensor) -> torch.tensor:
        x = x @ self.weight # Linear
        x = F.leaky_relu(x) # Activation
        return x 

class DeepNetwork(nn.Module):
    """Map 'dim' -vector ro a 'dim'-vector."""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(Block(dim) for i in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x 

def get_num_parameters(model):
    return sum(param.numel() for param in model.parameters())

def deep_network():
    # Define the network
    D = 8
    L = 3
    model = DeepNetwork(dim=D, num_layers = L).to(device)

    num_parameters = get_num_parameters(model)
    assert num_parameters == (D * D) * L

    B = 4
    x = torch.randn(B, D, device=device)
    print(model(x))

class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params:Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get squared gradients g2 = sum_{i<t} g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))# 创建与grad大小相同，位置相同的全零张量

                # Update optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2

                # Upadate parameters
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)


def optimizer():
    B = 2
    D = 4
    L = 3
    model = DeepNetwork(dim = D, num_layers = L).to(device)
    optimizer = AdaGrad(model.parameters(), lr = 0.01)

    # Compute gradients
    x = torch.randn(B, D, device = device)
    y = torch.tensor([4.,5.], device = device)
    pred_y = model(x).mean(dim = -1)
    print(pred_y, "\n")
    loss = F.mse_loss(input=pred_y, target=y)
    loss.backward()

    # Take a step
    optimizer.step()

    pre_y  = model(x).mean(dim = -1)
    print(pre_y,"\n")

    optimizer_state = {i: dict(p_state) for i, (p, p_state) in enumerate(optimizer.state.items())} 
    optimizer.zero_grad(set_to_none=True)
    
def train_loop():
    # True linear function with weights(0, 1, 2, ..., D-1)
    D = 16 # Dim size
    true_w = torch.arange(D, dtype = torch.float32, device = device)

    # Data loader that generates (x, y) pairs 
    B = 4 # Batch size 
    def get_batch() -> tuple[torch.tensor, torch.tensor]:
        x = torch.randn(B, D, device = device)
        true_y = x @ true_w # (B, 1)
        return (x, true_y)

    # Define the model and the optimizer
    L = 2
    model = DeepNetwork(dim = D, num_layers = L).to(device)
    optimizer = AdaGrad(model.parameters(), lr = 0.2)

    # Train !
    num_train_steps = 100
    epoch = 100
    for t in range(num_train_steps):
        # Get data
        x,y = get_batch()

        for i in range(epoch):
            # Forword
            pred_y = model(x).mean(dim=-1)

            # Backward
            loss = F.mse_loss(pred_y, y)
            loss.backward()

            # Optimizer
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
        

    # Test
    x,y = get_batch()
    pred_y = model(x).mean(dim=-1)
    print(f"x:\n{x}\npred_y:\n {pred_y}\ny:\n{y}")



if __name__ == "__main__":
    train_loop()