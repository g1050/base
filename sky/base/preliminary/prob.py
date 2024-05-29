import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones(6) / 6
print(fair_probs)
x = multinomial.Multinomial(1,fair_probs).sample()
print(x)
print(x.cumsum(dim=0))
print(multinomial.Multinomial(1000,fair_probs).sample().sum())
counts = multinomial.Multinomial(1000,fair_probs).sample() /1000
print(counts)
print('----')
# 查看收敛速度
# 模拟500次投掷
counts = multinomial.Multinomial(10,fair_probs).sample((500,))
print(counts)
cum_counts = counts.cumsum(dim=0)
print(cum_counts,cum_counts.shape)
estimates = cum_counts /cum_counts.sum(dim=1,keepdim=True)
print(estimates)
d2l.set_figsize((6,4.5))
for i in range(6):
    d2l.plt.plot(estimates[:,i].numpy(),label=(f"P{i+1}"))
    d2l.plt.axhline(y=0.167,color='black',linestyle='dashed')
    d2l.plt.gca().set_xlabel('Groups of experiments')
    d2l.plt.gca().set_ylabel('estimated probility')
d2l.plt.savefig("data/img.png")