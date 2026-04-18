import torch
import os

def normalized(p):
    p = p.float()
    return p / p.sum()
    
names = open(os.path.join("data", "names.txt"), "r").read().splitlines()

chars = sorted(list(set(''.join(names))))

char_index = {char:i+1 for i, char in enumerate(chars)}
index_char = {char_index[char]:char for _, char in enumerate(chars)}

char_index["."] = 0
index_char[0] = "."

counts = {}
values = sorted(counts.items(), key=lambda x: -x[1])
chars_c = len(chars) + 1

N = torch.zeros(chars_c, chars_c, dtype=torch.int32)

for name in names:
    chrs = '.' + name + '.'
    for c1, c2 in zip(chrs, chrs[1:]):
        idx1, idx2 = char_index[c1], char_index[c2]
        N[idx1, idx2] += 1
        
g = torch.Generator().manual_seed(2147483647)
P = normalized(N)
for i in range(10):
    ix = 0
    outs = []
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
        outs.append(index_char[ix])
        if ix == 0:
            print("".join(outs))
            break;


