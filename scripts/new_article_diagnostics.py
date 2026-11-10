"""Reproduce the v8 targeted experiment without silently reopening a final test."""
import argparse

from news_flywire.risk_v8 import prepare_data, development_baseline, train, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'baseline', 'train', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare_data, 'baseline': development_baseline,
     'train': train, 'evaluate': evaluate}[args.stage]()
