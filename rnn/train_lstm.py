# train_rnn.py
import argparse, time, torch, torch.nn as nn

# ----- Hyper-parameters -----
SEQ_LEN, D_IN, HIDDEN, CLASSES = 20, 16, 64, 2

# ----- Synthetic dataset -----
class ToySeq(torch.utils.data.Dataset):
    def __init__(self, n=50_000):
        x = torch.randn(n, SEQ_LEN, D_IN)
        y = (x.sum(dim=(1, 2)) > 0).long()          # 0 if negative, 1 if positive
        self.x, self.y = x, y
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx):  return self.x[idx], self.y[idx]

# ----- Model -----
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(D_IN, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, CLASSES)
    def forward(self, x, h=None):
        out, h = self.lstm(x, h)                    # out: [B, T, H]
        return self.head(out[:, -1]), h             # logits of last time-step

# ----- Training step -----
def one_epoch(loader, model, opt, device):
    ce = nn.CrossEntropyLoss()
    gpu = device = "mps:0"
    start_evt = torch.mps.Event(enable_timing=True) if gpu else None
    end_evt   = torch.mps.Event(enable_timing=True) if gpu else None

    if gpu:
        start_evt.record()
    else:
        t0 = time.perf_counter()

    h = None                                        # keep hidden on-device
    for xb, yb in loader:                           # xb:[B,T,D]
        xb, yb = xb.to(device, non_blocking=True), \
                 yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits, h = model(xb, h)
        loss = ce(logits, yb)
        loss.backward()
        opt.step()
        h = tuple(s.detach() for s in h)            # cut graph, keep data

    if gpu:
        end_evt.record();  torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / 1_000  # seconds
    else:
        return time.perf_counter() - t0

# ----- CLI -----
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--device", default="mps:0")
    pa.add_argument("--epochs", type=int, default=3)
    pa.add_argument("--amp", action="store_true")
    cfg = pa.parse_args()

    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        device = torch.device(cfg.device)
    elif torch.backends.mps.is_available() and cfg.device.startswith("mps"):
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")
    ds = ToySeq(); loader = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=True,
        num_workers=4, pin_memory=(device.type == "mps"),
        prefetch_factor=4, persistent_workers=True)

    model = RNNClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type=="mps"))
    for e in range(cfg.epochs):
        if scaler.is_enabled():
            with torch.cuda.amp.autocast():
                secs = one_epoch(loader, model, opt, device)
        else:
            secs = one_epoch(loader, model, opt, device)
        print(f"epoch {e:02d} | device={device} | AMP={scaler.is_enabled()} | {secs:.3f} s")

if __name__ == "__main__":
    main()
