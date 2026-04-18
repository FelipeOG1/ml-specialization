import torch
class NameTokenaizer:
    def __init__(self, names: list[str]):
        self._chars = sorted(list(set(''.join(names))))
        self._char_index = {char:index+1 for index, char in enumerate(self._chars)}
        self._index_char = {self._char_index[char]:char for _, char in enumerate(self._chars)}
        
        self._char_index["."] = 0
        self._index_char[0] = "."
        

    @property
    def num_chars(self):
        return len(self._chars) + 1
    
    def encode_char(self, char: str):
        return self._char_index[char]
    def decode_int(self, value: int):
        return self._index_char[value]
    
    def encode_string(self, name: str):
        return [self._char_index[char] for char in name]   
    def decode_list(self, encoded_str: list[int]):
        return ''.join([self._index_char[char] for char in encoded_str])

class DigramMap:
    def __init__(self, names):
        self._tokenaizer = NameTokenaizer(names)
        self._tensor = torch.zeros(self._tokenaizer.num_chars, self._tokenaizer.num_chars)
        self._generate_map()
        
    def _generate_map(self):
        for name in names:
            chrs = '.' + name + '.'
            for c1, c2 in zip(chrs, chrs[1:]):
                self._tensor[self._tokenaizer.encode_char(c1), self._tokenaizer.encode_char(c2)] +=1
        
    def __getitem__(self, index: int):
        return self._tensor[index]
           


    

if __name__ == "__main__":
    import os
    data = os.path.join("data", "names.txt")
    names = open(data, "r").read().splitlines()

    digram = DigramMap(names)
    print(digram[0])
