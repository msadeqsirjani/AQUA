"""
Utility script to extract results from PyTorch checkpoints and save as JSON/CSV.
Useful for converting existing .pt files to human-readable formats.
"""

import os
import json
import csv
import torch
import argparse


def extract_and_save_results(checkpoint_path, output_dir):
    """Extract results from .pt file and save as JSON and CSV."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract method and dataset from path
    parts = checkpoint_path.split('/')
    dataset = parts[-2] if len(parts) >= 2 else 'Unknown'
    method = parts[-3] if len(parts) >= 3 else 'Unknown'

    # Create summary
    summary = {
        'method': method,
        'dataset': dataset,
        'best_test_accuracy': float(checkpoint.get('best_accuracy', 0)),
        'energy_normalized': float(checkpoint.get('energy_normalized', 0)),
        'bitwidth': float(checkpoint.get('bitwidth', 0)),
    }

    # Add energy stats if available
    if 'energy_stats' in checkpoint:
        summary['energy_stats'] = checkpoint['energy_stats']

    # Save summary as JSON
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'✓ Saved summary to: {summary_path}')

    # Save per-epoch results as CSV if available
    if 'results' in checkpoint and checkpoint['results']:
        csv_path = os.path.join(output_dir, 'results.csv')
        results = checkpoint['results']

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'✓ Saved per-epoch results to: {csv_path}')

    # Print summary
    print('\n' + '='*60)
    print('EXTRACTED RESULTS:')
    print('='*60)
    print(json.dumps(summary, indent=2))
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract results from PyTorch checkpoint')
    parser.add_argument('checkpoint', help='Path to .pt checkpoint file')
    parser.add_argument('--output-dir', help='Output directory (default: same as checkpoint)', default=None)

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint)

    # Extract and save
    extract_and_save_results(args.checkpoint, output_dir)
