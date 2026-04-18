import torch
class NameTokenaizer:
    def __init__(self, names: list[str]):
        self._chars = sorted(list(set(''.join(names))))
        self._char_index = {char:index for index, char in enumerate(self._chars)}
        self._index_char = {self._char_index[char]:char for _, char in enumerate(self._chars)}
        
        self._char_index["."] = 0
        self._index_char[0] = "."
        

    @property
    def num_chars(self):
        return len(self._chars) + 1
    
    def encode(self, name: str):
        return [self._char_index[char] for char in name]
        
    def decode(self, encoded_str: list[int]):
        return ''.join([self._index_char[char] for char in encoded_str])

class DigramMap:
    def __init__(self, names):
        self._tokenaizer = NameTokenaizer(names)
        self._tensor = torch.zeros(self._tokenaizer.num_chars, self._tokenaizer.num_chars)
        

if __name__ == "__main__":
    import os
    data = os.path.join("data", "names.txt")
    names = open(data, "r").read().splitlines()

    digram = DigramMap(names)
