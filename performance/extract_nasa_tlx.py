#!/usr/bin/env python3
import os
import glob
import csv
import argparse
import sys
from statistics import mean
from collections import defaultdict
from tqdm import tqdm  # progress bar

def load_order_map(orders_path):
    # Read the experiment order file and create a map from participant ID to their condition order
    order_map = {}
    with open(orders_path, newline='', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';'])
        rdr = csv.DictReader(f, dialect=dialect)
        for row in rdr:
            order_map[row['ID']] = row['Order']
    return order_map

def parse_filename_and_condition(path, order_map):
    # Figure out which participant and condition a file belongs to, based on its name and the order map
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split('_')
    participant = parts[0]
    if participant not in order_map:
        return None, None
    try:
        block_idx = parts.index('block')
        block_num = int(parts[block_idx + 1])
    except (ValueError, IndexError):
        return None, None
    order_str = order_map[participant].replace('-', '')
    if block_num < 1 or block_num > len(order_str):
        return None, None
    condition = order_str[block_num - 1]
    return participant, condition

def read_tlx_from_file(path):
    # Read NASA-TLX scores from a file, looking for the right type of data
    tlx = {}
    with open(path, newline='', encoding='utf-8') as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';',' '])
        rdr = csv.DictReader(f, dialect=dialect)
        rdr.fieldnames = [h.strip() for h in rdr.fieldnames]
        for row in rdr:
            typ = row.get('type','').strip().lower()
            mod = row.get('module','').strip().lower()
            addr = row.get('address','').strip()
            val  = row.get('value','').strip()
            # Only keep rows that are NASA-TLX scores
            if typ == 'performance' and mod == 'genericscales':
                key = addr.lower().replace(' ', '_')
                try:
                    tlx[key] = float(val)
                except ValueError:
                    continue
    # Only return the scores if all required types are present
    required = {
        'mental_demand',
        'physical_demand',
        'time_pressure',
        'performance',
        'effort',
        'frustration'
    }
    if required.issubset(set(tlx.keys())):
        return {k: tlx[k] for k in required}
    else:
        return None

def main():
    # Set up command-line arguments for input/output files
    p = argparse.ArgumentParser(description="Extract NASA-TLX scores (genericscales) from MATB CSVs.")
    p.add_argument('input_dir', nargs='?', help="Folder containing *_block_*.csv files")
    p.add_argument('output_aggregated', nargs='?', help="Path to write the aggregated CSV")
    p.add_argument('output_average', nargs='?', help="Path to write the condition-averaged CSV")
    p.add_argument('--orders_csv', default='/Users/cartersale/Library/CloudStorage/OneDrive-MacquarieUniversity/Research/Projects/2025_MATBExp4/01_Experiment/exp4_orders.csv',
                   help="Path to exp4_orders.csv (default hardcoded)")
    args = p.parse_args()

    # If arguments aren't provided, ask the user for them
    if not args.input_dir:
        args.input_dir = input("Enter input directory with *_block_*.csv files: ").strip()
    if not args.output_aggregated:
        args.output_aggregated = input("Enter output path for aggregated CSV: ").strip()
    if not args.output_average:
        args.output_average = input("Enter output path for averaged-by-condition CSV: ").strip()

    # Make sure the input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Load the participant order information
    order_map = load_order_map(args.orders_csv)
    # Find all relevant CSV files in the input directory
    all_files = [
        f for f in glob.glob(os.path.join(args.input_dir, '*.csv'))
        if '_block_' in os.path.basename(f)
        and not f.endswith('_comms_log.csv')
    ]

    aggregated_rows = []

    # Go through each file and extract TLX scores
    for path in tqdm(all_files, desc='Parsing TLX Files', colour='green'):
        participant, condition = parse_filename_and_condition(path, order_map)
        if participant is None:
            continue
        tlx_data = read_tlx_from_file(path)
        if tlx_data is None:
            continue
        aggregated_rows.append({
            'participant': participant,
            'condition': condition,
            **tlx_data
        })

    if not aggregated_rows:
        print("No valid NASA-TLX data found.")
        return

    # Write all the extracted scores to a CSV file
    fieldnames = ['participant', 'condition',
                  'mental_demand', 'physical_demand', 'time_pressure',
                  'performance', 'effort', 'frustration']
    with open(args.output_aggregated, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow(row)
    print(f"✓ Wrote {len(aggregated_rows)} rows to: {args.output_aggregated}")

    # Group the scores by condition and calculate the average for each condition
    grouped = defaultdict(list)
    for row in aggregated_rows:
        grouped[row['condition']].append(row)

    average_rows = []
    for cond, group in grouped.items():
        avg_row = {'condition': cond}
        for key in ['mental_demand', 'physical_demand', 'time_pressure',
                    'performance', 'effort', 'frustration']:
            avg_row[key] = mean([r[key] for r in group])
        average_rows.append(avg_row)

    # Sort the results by condition and write to another CSV file
    average_rows.sort(key=lambda x: x['condition'])

    with open(args.output_average, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition'] + fieldnames[2:])
        writer.writeheader()
        for row in average_rows:
            writer.writerow(row)
    print(f"✓ Wrote {len(average_rows)} averaged rows to: {args.output_average}")

if __name__ == '__main__':
    main()
