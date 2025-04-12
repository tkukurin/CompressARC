"""Convert [agi2] to single file same as with this dataset.

TODO check, untested, I think this is how they convert.

```
cd dataset
git clone git@github.com:arcprize/ARC-AGI-2.git
python agi2_to_single.py ARC-AGI-2/data/evaluation .
python agi2_to_single.py ARC-AGI-2/data/training .
```

[agi2]: https://github.com/arcprize/ARC-AGI-2
"""
import os
import json
import argparse
from glob import glob

def main():
    parser = argparse.ArgumentParser(description='ARC-AGI2 to CompressARC format.')
    parser.add_argument('input_dir', help='Directory containing JSON files')
    parser.add_argument('output_dir', help='Output dir', default=None)
    args = parser.parse_args()

    split = "evaluation" if "evaluation" in args.input_dir else "training"
    outdir = args.output_dir or os.path.dirname(args.input_dir)
    
    challenges = {}; solutions = {}
    for json_file in glob(os.path.join(args.input_dir, '*.json')):
        key = os.path.splitext(os.path.basename(json_file))[0]
        with open(json_file, 'r') as f:
            data = json.load(f)
            last = data['test'].pop()
            last_out = last.pop('output')
            data['test'].append(last)
            challenges[key] = data
            solutions[key] = [last_out]

    chpath = os.path.join(outdir, f'arc-agi2_{split}_challenges.json')
    with open(chpath, 'w') as f:  # separators to reduce spacing
        json.dump(challenges, f, indent=None, separators=(',', ':'))
    solpath = os.path.join(outdir, f'arc-agi2_{split}_solutions.json')
    with open(solpath, 'w') as f:  # separators to reduce spacing
        json.dump(solutions, f, indent=None, separators=(',', ':'))
    
    print(f"Converted {len(challenges)} files from {args.input_dir} to {chpath} and {solpath}")

if __name__ == '__main__': main()