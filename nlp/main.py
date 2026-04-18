import torch
import os
names = open(os.path.join("data", "names.txt"), "r").read().splitlines()

chars = sorted(list(set("".join(names))))

char_index = {char:i+1 for i, char in enumerate(chars)}
index_char = {char_index[char]:char for _, char in enumerate(chars)}

char_index["."] = 0
index_char[0] = "."
print(char_index)

counts = {}
for name in names:
    chs = ['.'] + list(name) + ['.']
    for c1, c2 in zip(chs, chs[1:]):
        bigram = (char_index[c1], char_index[c2])
        counts[bigram] = counts.get(bigram, 0) + 1

values = sorted(counts.items(), key=lambda x: -x[1])
chars_c = len(chars) + 1

a = torch.zeros(chars_c, chars_c, dtype=torch.int32)

for name in names:
    for c1, c2 in zip(name, name[1:]):
        idx1, idx2 = char_index[c1], char_index[c2]
        a[idx1, idx2] += 1




