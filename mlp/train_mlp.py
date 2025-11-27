# train_mlp.py

import argparse, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- synthetic task ----------
def make_dataset(n = 60_000, d_in = 100, classes = 10):
    X = torch.randn(n, d_in)
    y = torch.randint(0, classes, (n,))
    return TensorDataset(X, y)

class MLP(nn.Module):
    def __init__(self, d_in = 100, hidden = 256, classes = 10):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(d_in, hidden), 
                nn.ReLU(),
                nn.Linear(hidden, classes))

    def forward(self, x): 
        return self.net(x)

class Trainer:
    def __init__(self, model: torch.nn.Module, criterion: callable, optimizer: torch.optim,
                 train_iter: torch.utils.data.TensorDataset,
                 par):
        self.model = model
        self.criterion = criterion
        self.opt = optimizer
        self.train_iter = train_iter
        self.par = par

    def train(self):

        lossList = []

        for xb, yb in self.train_iter: 
            xb = xb.to(self.par.device, non_blocking = True)
            yb = yb.to(self.par.device, non_blocking = True)          
        self.opt.zero_grad(set_to_none = True)
        loss = self.criterion(self.model(xb), yb)                 
        loss.backward()
        self.opt.step()

        lossList.append(loss.item())

        return lossList
    
    def run(self):
        
        for e in range(self.par.epochs):

            if self.par.device == "cuda:0":
                start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_evt.record()
            elif self.par.device == "mps:0":
                start_evt, end_evt = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
                start_evt.record()
            else:
                t0 = time.perf_counter()

            loss = self.train()

            if self.par.device == "cuda:0":
                end_evt.record();  torch.mps.synchronize()
                secs = start_evt.elapsed_time(end_evt) / 1_000
            if self.par.device == "mps:0":
                end_evt.record();  torch.mps.synchronize()
                secs = start_evt.elapsed_time(end_evt) / 1_000
            else:
                secs = time.perf_counter() - t0

            print(f"Epoch {e:02d}   device={self.par.device}   time={secs:.3f}s")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps:0", help="cuda:0 or mps:0 or cpu")
    p.add_argument("--epochs", type=int, default = 10)
    args = p.parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    elif torch.backends.mps.is_available() and args.device.startswith("mps"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    
    print(device)

    ds = make_dataset()
    loader = DataLoader(ds, batch_size=20480, shuffle=True,
                   num_workers=4, pin_memory=(device.type in ['cuda', 'mps']),
                   prefetch_factor=4, persistent_workers=True)
    model= MLP().to(device) 
    opt = torch.optim.SGD(MLP().parameters(), lr = 0.1)
    
    trainer = Trainer(criterion = nn.CrossEntropyLoss(),
                      model = model,
                      optimizer = opt,
                      train_iter = loader,
                      par = args)
    trainer.run()

if __name__ == "__main__":
    main()






