#!/usr/bin/env python3
"""
Entry point.
  python run.py                    # full pipeline
  python run.py --stage signal     # Stage A only
  python run.py --stage modeling   # Stage B only
  python run.py --stage scaling    # Stage C only
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline, run_signal_stage, run_modeling_stage, run_scaling_stage
from src.utils import load_config

def main():
    p = argparse.ArgumentParser(description="Membrane aging prediction pipeline")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--stage", type=str, default="all",
                   choices=["all", "signal", "modeling", "scaling"])
    p.add_argument("--raw-file", type=str, default=None, help="Path to raw CSV")
    p.add_argument("--aging-file", type=str, default=None, help="Path to aging CSV")
    args = p.parse_args()

    if args.stage == "all":
        run_pipeline(args.config)
    else:
        cfg = load_config(args.config)
        if args.stage == "signal" and args.raw_file:
            run_signal_stage(cfg, args.raw_file)
        elif args.stage == "modeling" and args.aging_file:
            run_modeling_stage(cfg, args.aging_file)
        elif args.stage == "scaling" and args.aging_file:
            run_scaling_stage(cfg, args.aging_file)
        else:
            print(f"Stage '{args.stage}' requires --raw-file or --aging-file")

if __name__ == "__main__":
    main()
