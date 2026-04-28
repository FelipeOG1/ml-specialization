from main import NameTokenaizer
import os
import torch
import torch.nn.functional as F
path = os.path.join('data','names.txt')
names = open(path, 'r').read().splitlines()
tok = NameTokenaizer(names)

def get_training_set() -> torch.tensor:
    x, y = [], []
    for name in names:
        name = tok.prepare_name(name)
        for c1, c2 in zip(name, name[1:]):
            x.append(tok.encode_char(c1))
            y.append(tok.encode_char(c2))
    return torch.tensor(x), torch.tensor(y)


class Model:
    def __init__(self, w: torch.tensor):
        self.w = w
    def forward(self, x: torch.tensor):      
        counts = (x @ self.w).exp()
        return counts / counts.sum(1, keepdim=True)

x, y = get_training_set()

g = torch.Generator().manual_seed(2147483647)
w = torch.rand((27, 27), generator=g, requires_grad=True)
model = Model(w)

one = F.one_hot(x, num_classes=27).float()


n_values = 10

for _ in range(300):
    pred = model.forward(one)
    w.grad = None
    loss = -pred[torch.arange(pred.shape[0]), y].log().mean()
    loss.backward()
    w.data += -50 * w.grad
    print(loss)

