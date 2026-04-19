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
    
    def prepare_name(self, name: str):
        return '.' + name + '.'
        
    def encode_char(self, char: str):
        return self._char_index[char]
    def decode_int(self, value: int):
        return self._index_char[value]
    
    def encode_string(self, name: str):
        return [self._char_index[char] for char in name]   
    def decode_list(self, encoded_str: list[int]):
        return ''.join([self._index_char[char] for char in encoded_str])

class DigramModel:
    def __init__(self, names):
        self._tokenaizer = NameTokenaizer(names)
        self._map = torch.zeros(self._tokenaizer.num_chars, self._tokenaizer.num_chars)
        self._generate_map()
        self.P = self._map / self._map.float().sum(1, keepdim=True)
    def _generate_map(self):
        for name in names:
            name = self._tokenaizer.prepare_name(name) 
            for c1, c2 in zip(name, name[1:]):
                self._map[self._tokenaizer.encode_char(c1), self._tokenaizer.encode_char(c2)] +=1
        
    def __getitem__(self, idx):
        return self._map[idx]


    def get_loss_name(self, name):
        log_lh = 0.0
        iterations = 0
        name = self._tokenaizer.prepare_name(name) 
        for c1, c2 in zip(name, name[1:]):
            log_prob = torch.log(self.P[self._tokenaizer.encode_char(c1), self._tokenaizer.encode_char(c2)])
            log_lh += log_prob
            iterations += 1
        return -(log_lh / iterations)
if __name__ == "__main__":
    import os
    data = os.path.join("data", "names.txt")
    names = open(data, "r").read().splitlines()
    m = DigramModel(names)
    print(m.get_loss_name("emmanuel"))
    
