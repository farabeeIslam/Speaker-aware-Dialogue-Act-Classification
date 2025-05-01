# run_t5_swda.py

import argparse
from transformers import T5Tokenizer
from engine_t5_swda import EngineT5SWDA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='swda')          # dataset name
    parser.add_argument('--mode', type=str, default='train')           # train or test
    parser.add_argument('--nclass', type=int, default=43)              # number of dialogue act classes
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batch_size_val', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)               # keep 4 for fast debug, increase later
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='0')                # GPU index
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    engine = EngineT5SWDA(args, tokenizer)
    engine.train()

if __name__ == '__main__':
    main()







