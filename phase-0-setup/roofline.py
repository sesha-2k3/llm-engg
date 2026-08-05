#!/usr/bin/env python3

"""
Decode-throughput ceiling from memory bandwidth.
 
During autoregressive decode at batch size 1, generating ONE token requires
reading the entire model's weights from memory. So:
 
    max tokens/sec  =  memory bandwidth (GB/s)  /  model size (GB)
 
That's the whole idea. Predict before you measure; the gap between prediction
and reality is what you're actually studying.
 
Examples:
    python roofline.py --size-gb 4.7 --chip m4
    python roofline.py --size-gb 4.7 --chip m4 --measured 19.4
    python roofline.py --size-gb 4.7 --all
"""
 
import argparse
 
# GB/s
CHIPS = {
    "m4":       ("Apple M4 (16GB unified)",        120),
    "m4pro":    ("Apple M4 Pro",                   273),
    "t4":       ("NVIDIA T4 (Kaggle)",             320),
    "p100":     ("NVIDIA P100 (Kaggle)",           732),
    "l4":       ("NVIDIA L4",                      300),
    "a100":     ("NVIDIA A100 40GB",              1555),
    "h100":     ("NVIDIA H100 SXM",               3350),
}
 
 
def predict(size_gb: float, bw: float) -> float:
    return bw / size_gb
 
 
def report(label: str, bw: float, size_gb: float, measured: float | None) -> None:
    ceiling = predict(size_gb, bw)
    line = f"{label:<32} {bw:>7.0f} GB/s   ceiling {ceiling:>7.1f} tok/s"
    if measured is not None:
        eff = 100.0 * measured / ceiling
        line += f"   measured {measured:>6.1f}   efficiency {eff:>5.1f}%"
    print(line)
 
 
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size-gb", type=float, required=True,
                   help="on-disk / in-memory size of the weights in GB (e.g. 4.7 for an 8B at Q4_K_M)")
    p.add_argument("--chip", choices=sorted(CHIPS), help="preset chip")
    p.add_argument("--bandwidth", type=float, help="custom bandwidth in GB/s")
    p.add_argument("--measured", type=float, help="your measured decode tok/s, to compute efficiency")
    p.add_argument("--all", action="store_true", help="show every preset chip")
    args = p.parse_args()
 
    print(f"\nWeights: {args.size_gb} GB\n")
 
    if args.all:
        for key in sorted(CHIPS, key=lambda k: CHIPS[k][1]):
            label, bw = CHIPS[key]
            report(label, bw, args.size_gb, args.measured if key == args.chip else None)
    elif args.bandwidth:
        report("custom", args.bandwidth, args.size_gb, args.measured)
    elif args.chip:
        label, bw = CHIPS[args.chip]
        report(label, bw, args.size_gb, args.measured)
    else:
        p.error("pass --chip, --bandwidth, or --all")
 
    print("\nNote: real efficiency of 60-85% is normal. Below that, look for "
          "CPU-side overhead, unfused ops, or a bad quantization kernel.\n")
 
 
if __name__ == "__main__":
    main()