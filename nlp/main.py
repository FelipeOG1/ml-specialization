import torch
import os
names = open(os.path.join("data", "names.txt"), "r").read().splitlines()

chars = list(set("".join(names)))

char_index = {char:i for i, char in enumerate(chars)}
index_char = {char_index[char]:char for _, char in enumerate(chars)}


encoder = lambda text: [char_index[char] for char in text]
decoder = lambda array: "".join([index_char[index] for index in array])
pairs = lambda name: zip(name, name[1:])


counts = {}
for name in names:
    chs = ['<S>'] + list(name) + ['<E>']
    for c1, c2 in zip(chs, chs[1:]):
        bigram = (c1, c2)
        counts[bigram] = counts.get(bigram, 0) + 1

values = sorted(counts.items(), key=lambda x:-x[1])

print(values[:10])
