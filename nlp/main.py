
class NameTokenaizer:
    def __init__(self, names: list[str]):
        _chars = sorted(list(set(''.join(names))))
        
        self._char_index = {char:index for index, char in enumerate(_chars)}
        self._index_char = {self._char_index[char]:char for _, char in enumerate(_chars)}
    def encode(self, name: str):
        return [self._char_index[char] for char in name]
        
    def decode(self, encoded_str: list[int]):
        return ''.join([self._index_char[char] for char in encoded_str])


if __name__ == "__main__":
    import os
    data = os.path.join("data", "names.txt")
    names = open(data, "r").read().splitlines()

