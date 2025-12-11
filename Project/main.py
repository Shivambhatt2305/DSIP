# main.py
import argparse
import os
from optimize import optimize_for_prompt
from utils import save_wav, plot_spectrogram

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt (e.g. 'bees buzzing')")
    parser.add_argument('--gens', type=int, default=150, help="Generations (more = better quality, slower)")
    parser.add_argument('--pop', type=int, default=30, help="Population size")
    parser.add_argument('--dur', type=float, default=2.0, help="Sound duration (seconds)")
    parser.add_argument('--outdir', type=str, default='outputs', help="Output directory")
    parser.add_argument('--out', type=str, help="Optional: Specific output filename (will be placed in outdir)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("Prompt:", args.prompt)
    best_params, best_audio, history = optimize_for_prompt(args.prompt, gens=args.gens, pop_size=args.pop, dur=args.dur)
    # save
    if args.out:
        outwav = os.path.join(args.outdir, args.out)
    else:
        outwav = f"{args.outdir}/result_{args.prompt.replace(' ','_')[:60]}.wav"
    save_wav(outwav, best_audio)
    outspec = f"{args.outdir}/spec_{args.prompt.replace(' ','_')[:60]}.png"
    plot_spectrogram(best_audio, outpath=outspec, title=f"Result: {args.prompt}")
    # print final params
    print("Best params:")
    for k,v in best_params.items():
        print(f"  {k}: {v}")
    # save params to text
    with open(f"{args.outdir}/params_{args.prompt.replace(' ','_')[:60]}.txt","w") as f:
        for k,v in best_params.items():
            f.write(f"{k}: {v}\n")
    print("Done. Outputs in", args.outdir)

if __name__ == "__main__":
    main()
