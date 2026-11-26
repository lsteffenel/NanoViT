import argparse
import csv
import glob
import json
import os


def parse_runs(outputs_dir):
    runs = {}
    for log in sorted(glob.glob(os.path.join(outputs_dir, "*", "log.txt"))):
        run_dir = os.path.dirname(log)
        best = None
        for line in open(log, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                stats = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "test_acc1" not in stats:
                continue
            if best is None or stats["test_acc1"] > best["test_acc1"]:
                best = stats
        if best is not None:
            runs[os.path.basename(run_dir)] = {
                "acc1": best["test_acc1"],
                "acc5": best.get("test_acc5", float("nan")),
                "epoch": best.get("epoch"),
                "n_parameters": best.get("n_parameters"),
            }
    return runs


def parse_bench(csv_path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["model"]] = row
    return rows


def latest_bench(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "bench_*.csv")))
    return files[-1] if files else None


def main():
    p = argparse.ArgumentParser(description="Agrege log.txt (runs) + bench CSV -> table markdown")
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--bench_csv", default=None, help="defaut: dernier phase0/results/bench_*.csv")
    p.add_argument("--out", default=None, help="fichier markdown de sortie (defaut: stdout)")
    args = p.parse_args()

    runs = parse_runs(args.outputs_dir)
    bench_csv = args.bench_csv or latest_bench(os.path.join("phase0", "results"))
    bench = parse_bench(bench_csv) if bench_csv and os.path.exists(bench_csv) else {}

    header = ("| run | modele | acc1 (%) | acc5 (%) | params (M) | MAdds reparam (G) "
              "| lat GPU bs1 (ms/img) | lat CPU bs1 (ms/img) | bench_csv |")
    sep = "|" + "---|" * 8
    lines = [header, sep]

    def model_of(run_name):
        for m in bench:
            if m.lower() in run_name.lower():
                return m
        for m in ("MobileViTv4_linear_XS", "MobileViTv4_linear_XXS", "MobileViTv4_flat_XXS",
                  "MobileViTv4_XXS", "MobileViTv4_XS", "MobileViTv4_S", "MobileViTv4_M",
                  "MobileViTv4_L"):
            if m.lower() in run_name.lower():
                return m
        return run_name

    rows = []
    for run_name, r in sorted(runs.items()):
        m = model_of(run_name)
        b = bench.get(m, {})
        rows.append((
            float(b.get("madds_reparam_G") or "nan"),
            f"| {run_name} | {m} | {r['acc1']:.2f} | {r['acc5']:.2f} "
            f"| {b.get('params_reparam_M', 'nan')} | {b.get('madds_reparam_G', 'nan')} "
            f"| {b.get('lat_dev_bs1_ms_per_img', 'nan')} | {b.get('lat_cpu_bs1_ms_per_img', 'nan')} "
            f"| {os.path.basename(bench_csv) if bench_csv else '-'} |",
        ))

    rows.sort(key=lambda t: t[0])
    for _, line in rows:
        lines.append(line)

    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nEcrit dans : {args.out}")


if __name__ == "__main__":
    main()
