import torch
import torch.nn.functional as F
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
        
    def encode_char(self, char: str) -> int:
        return self._char_index[char]
    
    def decode_int(self, value: int):
        return self._index_char[value]
    
    def encode_string(self, name: str):
        return [self._char_index[char] for char in name]
    def decode_list(self, encoded_str: list[int]):
        return ''.join([self._index_char[char] for char in encoded_str])
    
    def decode_int(self, value: int):
        return self._index_char[value]
    
class Digram:
    def __init__(self, names):
        self._data = names
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

    def get_training_set(self) -> tuple[torch.tensor, torch.tensor]:
        x, y = [], []
        for name in self._data:
            name = self._tokenaizer.prepare_name(name)
            for c1, c2 in zip(name, name[1:]):    
                x.append(self._tokenaizer.encode_char(c1)) 
                y.append(self._tokenaizer.encode_char(c2))
            

        return torch.tensor(x), torch.tensor(y)
    
class Model:
    def __init__(self, w: torch.tensor,
                 tokenaizer: NameTokenaizer) -> None:
        self.w = w
        self.tok = tokenaizer
        
    def __call__(self, x: torch.tensor):
        counts = (x @ self.w).exp()
        return counts / counts.sum(1, keepdims=True)
    

    def predict_char(self, ch: str) -> str:
        encoded_char = self.tok.encode_string(ch)
        x = F.one_hot(torch.tensor(encoded_char), num_classes=27).float()
        return self(x)

    def _get_loss(self, y_hat: torch.tensor, y: torch.tensor):
        loss = -y_hat[torch.arange(y_hat.shape[0]), y].log().mean()
        return loss
    
    
    def fit(self, x: torch.tensor, y: torch.tensor, 
            epochs=100, learning_rate=0.1,
            ) -> torch.tensor:
        eps = 1e-6
        loss = None
        for i in range(epochs):
            prev_loss = self._get_loss(self(x), y)
            self.w.grad = None
            prev_loss.backward()
            self.w.data += -learning_rate * self.w.grad 
            new_loss = self._get_loss(self(x), y)
            loss = new_loss
            print(loss.item())
            if torch.abs(prev_loss - new_loss) < eps:
                print(f'convergence at {i} iteration')
                break

        return self.w
    
    def set_w(self, w: torch.tensor) -> None : self.w = w
    
if __name__ == "__main__":
    import os
    data = os.path.join("data", "names.txt")
    names = open(data, "r").read().splitlines()
    g = torch.Generator().manual_seed(2147483647)
    digram = Digram(names)
    x, y = digram.get_training_set()
    x = F.one_hot(x, num_classes=27).float()
    W = torch.rand((27, 27), generator=g, requires_grad=True)
    tok = NameTokenaizer(names)

    m = Model(W, tokenaizer=tok)
    w = m.fit(x, y, epochs=500, learning_rate=30)
    
       
   
          


    
