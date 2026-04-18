import torch
import os

    
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
p = N[0].float()
p = p / p.sum()
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
P = (N + 1).float()
P /= P.sum(1, keepdim=True)

like_hood = 0.0
n_names = 0
for name in ["andrejq"]:
    chrs = '.' + name + '.'
    for ch1, ch2 in zip(chrs, chrs[1:]):
        idx1, idx2 = char_index[ch1], char_index[ch2]
        prob = P[idx1, idx2]
        log_prob = torch.log(prob)
        like_hood += log_prob
        n_names += 1
neg_like = -(like_hood)
nll = neg_like / n_names
print(nll)
