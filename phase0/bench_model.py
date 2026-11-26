import argparse
import copy
import csv
import os
import sys
import time
from types import SimpleNamespace

import torch
from timm.models import create_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import models.nanovit  # noqa: F401  (enregistre les modeles NanoViT dans timm)

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

CSV_FIELDS = [
    "model", "input_size", "num_classes",
    "params_M", "madds_train_G",
    "params_reparam_M", "madds_reparam_G",
    "lat_dev_bs1_ms_per_img", "lat_dev_bsB_ms_per_img", "lat_cpu_bs1_ms_per_img",
    "acc1", "acc5", "loss", "ckpt",
]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_madds(model, input_size, device):
    if FlopCountAnalysis is None:
        return float("nan")
    x = torch.ones((1, 3, input_size, input_size), device=device)
    return FlopCountAnalysis(model, x).total()


def reparameterize(model):
    m = copy.deepcopy(model)
    m.eval()
    for module in m.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return m


def measure_latency(model, x, iters, warmup, device):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / iters * 1000.0 / x.size(0)  # ms par image


def build_val_loader(args, nb_classes):
    from datasets import build_dataset
    ns = SimpleNamespace(
        input_size=args.input_size, crop_pct=None,
        imagenet_default_mean_and_std=True,
        data_set=args.data_set, data_path=args.data_path,
        eval_data_path=args.eval_data_path, nb_classes=nb_classes,
        multi_scale_sampler=False,
        color_jitter=0.4, aa="rand-m9-mstd0.5-inc1",
        train_interpolation="bicubic", reprob=0.0, remode="pixel",
        recount=1, three_aug=False,
    )
    dataset, nb_classes = build_dataset(is_train=False, args=ns)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return loader, nb_classes


@torch.no_grad()
def evaluate(model, loader, device):
    from timm.utils import accuracy
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    sum1 = sum5 = sumloss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        sumloss += criterion(out, y).item() * x.size(0)
        acc1, acc5 = accuracy(out, y, topk=(1, 5))
        sum1 += acc1.item() * x.size(0)
        sum5 += acc5.item() * x.size(0)
        n += x.size(0)
    return sum1 / n, sum5 / n, sumloss / n


def load_ckpt(model, ckpt):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


def infer_num_classes(ckpt):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return int(sd["head.weight"].shape[0])


def main():
    p = argparse.ArgumentParser(description="Benchmark NanoViT: params, MAdds (train/reparam), latence, acc.")
    p.add_argument("--model", required=True)
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--ckpt", default=None, help="checkpoint 'model' a charger (optionnel)")
    p.add_argument("--data_set", default="IMNETTE")
    p.add_argument("--data_path", default="./data/imagenette")
    p.add_argument("--eval_data_path", default=None)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bench_batch", type=int, default=256, help="batch pour la latence sur --device")
    p.add_argument("--bench_iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--no_cpu_bench", action="store_true")
    p.add_argument("--csv", default=None, help="fichier CSV de sortie (append)")
    args = p.parse_args()

    device = torch.device(args.device)

    num_classes = args.num_classes
    if num_classes is None and args.ckpt:
        num_classes = infer_num_classes(args.ckpt)
    if num_classes is None:
        loader, num_classes = build_val_loader(args, 1000)
        args.num_classes = num_classes

    model = create_model(args.model, img_size=args.input_size, pretrained=False,
                         num_classes=num_classes)
    if model.head.out_features != num_classes:
        model.head = torch.nn.Linear(model.head.in_features, num_classes, bias=False)

    if args.ckpt:
        load_ckpt(model, args.ckpt)

    model = model.to(device).eval()

    params = count_params(model)
    madds_train = compute_madds(model, args.input_size, device)

    model_rep = reparameterize(model).to(device)
    params_rep = count_params(model_rep)
    madds_rep = compute_madds(model_rep, args.input_size, device)

    x1 = torch.randn(1, 3, args.input_size, args.input_size, device=device)
    xB = torch.randn(args.bench_batch, 3, args.input_size, args.input_size, device=device)
    lat1 = measure_latency(model_rep, x1, args.bench_iters, args.warmup, device)
    latB = measure_latency(model_rep, xB, max(1, args.bench_iters // 4), args.warmup, device)

    lat_cpu = float("nan")
    if not args.no_cpu_bench:
        model_cpu = reparameterize(model).to("cpu")
        xc = torch.randn(1, 3, args.input_size, args.input_size, device="cpu")
        lat_cpu = measure_latency(model_cpu, xc, max(5, args.bench_iters // 10), args.warmup,
                                  torch.device("cpu"))

    acc1 = acc5 = loss = float("nan")
    if args.ckpt:
        if args.num_classes is None:
            args.num_classes = num_classes
        loader, _ = build_val_loader(args, num_classes)
        acc1, acc5, loss = evaluate(model_rep, loader, device)

    row = {
        "model": args.model, "input_size": args.input_size, "num_classes": num_classes,
        "params_M": round(params * 1e-6, 3), "madds_train_G": round(madds_train * 1e-9, 3),
        "params_reparam_M": round(params_rep * 1e-6, 3), "madds_reparam_G": round(madds_rep * 1e-9, 3),
        "lat_dev_bs1_ms_per_img": round(lat1, 4), "lat_dev_bsB_ms_per_img": round(latB, 4),
        "lat_cpu_bs1_ms_per_img": round(lat_cpu, 4) if lat_cpu == lat_cpu else "nan",
        "acc1": round(acc1, 3) if acc1 == acc1 else "nan",
        "acc5": round(acc5, 3) if acc5 == acc5 else "nan",
        "loss": round(loss, 4) if loss == loss else "nan",
        "ckpt": args.ckpt or "",
    }
    print(row)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
