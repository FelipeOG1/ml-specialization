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
    
            
class TrainingSet:
    def __init__(self, names: list[str],
                 tokenaizer: NameTokenaizer):
        self.names = names
        self._tok = tokenaizer
        
    def get_training_set(self, context_size: int) -> tuple[torch.tensor, torch.tensor]:
        x, y = [], []
        for name in self.names:
            context: list[int] = [0] * context_size
            for char in name + '.':
                ix = self._tok.encode_char(char)
                x.append(context)
                y.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(x), torch.tensor(y)
class Model:
    def __init__(self, w: torch.tensor,
                 tokenaizer: NameTokenaizer) -> None:
        self.w = w
        self.tok = tokenaizer
        self._g = torch.Generator().manual_seed(2147483647)
   
    def __call__(self, x: torch.tensor):
        counts = (x @ self.w).exp()
        return counts / counts.sum(1, keepdims=True)
    
    def predict_char(self, ch: str) -> str:
        encoded_char = self.tok.encode_string(ch)
        x = F.one_hot(torch.tensor(encoded_char), num_classes=27).float()
        prediction = self(x)
        return self.tok.decode_int(torch.argmax(prediction).item())
    
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
            
            if loss.item() < 2.48: break
            
            if torch.abs(prev_loss - new_loss).item() < eps:
                print(f'convergence at {i} iteration')
                break

        return self.w
    
    def set_w(self, w: torch.tensor) -> None : self.w = w


    def create_names(self, n_names=5):
        for _ in range(n_names):
            out = []
            ix = 0
            while True:
                x = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                p = self(x)
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=self._g).item()
                out.append(ix)
                if ix == 0:break
                
            name: str = self.tok.decode_list(out)
            print(name)
if __name__ == "__main__":
    import os
    data = os.path.join("data", "names.txt")
    names = open(data, "r").read().splitlines()
    g = torch.Generator().manual_seed(2147483647)
    tok = NameTokenaizer(names)
    x, y = TrainingSet(names, tok).get_training_set(context_size=3)
    print(x)
