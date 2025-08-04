# This script calculates accuracy and response times for different tasks in the MATB experiment.
# It reads event logs, scores performance, and saves results to a CSV file.

#!/usr/bin/env python3
import os, sys, glob, csv, argparse
from datetime import datetime
from statistics import mean
from collections import defaultdict, Counter
from tqdm import tqdm

# Load the experiment order file, which tells us the condition order for each participant
orders_path = 'extras/exp4_orders.csv'
order_map = {}
with open(orders_path, newline='', encoding='utf-8-sig') as f:
    sample = f.read(2048)
    f.seek(0)
    dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';'])
    rdr = csv.DictReader(f, dialect=dialect)
    for row in rdr:
        order_map[row['ID']] = row['Order']

# Define which event types and addresses we care about for scoring
TRACK_ADDR   = {"cursor_in_target"}
RESMAN_ADDR  = {"a_in_tolerance", "b_in_tolerance"}
SYSMON_ADDR  = {"signal_detection","response_time"}
VALID_MODS   = {"track", "resman", "sysmon", "communications"}
ADDRS        = TRACK_ADDR | RESMAN_ADDR | SYSMON_ADDR

def read_csv_events(path):
    # Read a CSV file and extract performance, prompt, and keypress events
    parts = os.path.splitext(os.path.basename(path))[0].split('_')
    dt0 = datetime.strptime(parts[-2] + parts[-1], '%y%m%d%H%M%S')
    perf, prompts, keys = [], [], []
    with open(path, newline='', encoding='utf-8') as f:
        sample = f.read(2048); f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';',' '])
        rdr = csv.DictReader(f, dialect=dialect)
        rdr.fieldnames = [h.strip() for h in rdr.fieldnames]
        for row in rdr:
            typ = row['type'].strip().lower()
            mod = row['module'].strip().lower()
            addr= row['address'].strip()
            t   = float(row.get('scenario_time', 0))
            val = row['value'].strip()
            # Performance events for scoring
            if typ=='performance' and mod in VALID_MODS and addr in ADDRS:
                perf.append({'mod':mod, 'addr':addr, 'val':val, 't':t})
            # Communication prompts (own/other)
            if typ=='event' and mod=='communications' and addr=='radioprompt':
                if val in ('own','other'):
                    prompts.append({'t':t, 'kind':val})
            # Keypress events (spacebar or joystick button)
            if typ=='input' and (
                (mod=='keyboard' and addr=='SPACE') or
                (mod=='joystick' and addr.startswith('JOY_BTN_'))
            ) and val in ('press','release'):
                keys.append({'t': t, 'action': val})
    return dt0, perf, prompts, keys

def pair_keys(keys):
    # Match key press and release events to measure how long a key was held down
    keys = sorted(keys, key=lambda x: x['t'])
    pairs = []
    i = 0
    while i < len(keys):
        if keys[i]['action']=='press':
            for j in range(i+1, len(keys)):
                if keys[j]['action']=='release':
                    dur = keys[j]['t'] - keys[i]['t']
                    if dur >= 1.5:
                        pairs.append({
                            't_press': keys[i]['t'],
                            't_release': keys[j]['t'],
                            'dur': dur,
                            'matched': False
                        })
                    i = j
                    break
        i += 1
    return pairs

def compute_point_metrics(perf, prompts, keys):
    # Calculate accuracy and response times for each task

    # Tracking: % of time cursor was in the target
    cursor_vals = [int(r['val']) for r in perf if r['mod']=='track' and r['addr']=='cursor_in_target']
    track_score = sum(cursor_vals)
    track_total = len(cursor_vals)
    track_point_accuracy = track_score / track_total if track_total else 0.0

    # Resource Management: % of time both resources were in tolerance
    a_vals = [int(r['val']) for r in perf if r['mod']=='resman' and r['addr']=='a_in_tolerance']
    b_vals = [int(r['val']) for r in perf if r['mod']=='resman' and r['addr']=='b_in_tolerance']
    resman_score = sum(a_vals) + sum(b_vals)
    resman_total = len(a_vals) + len(b_vals)
    resman_point_accuracy = resman_score / resman_total if resman_total else 0.0

    # System Monitoring: hits minus false alarms, divided by total
    sysmon_labels = [r['val'].lower() for r in perf if r['mod']=='sysmon' and r['addr']=='signal_detection']
    hits = sysmon_labels.count('hit')
    fas  = sysmon_labels.count('fa')
    sysmon_score = hits - fas
    sysmon_total = len(sysmon_labels)
    sysmon_point_accuracy = sysmon_score / sysmon_total if sysmon_total else 0.0

    # System Monitoring: average reaction time
    rt_vals = [float(r['val'])
            for r in perf
            if r['mod']=='sysmon'
            and r['addr']=='response_time'
            and r['val'].replace('.','',1).isdigit()]
    mean_sysmon_rt = mean(rt_vals) if rt_vals else 0.0

    # Communications: score and response time based on prompts and keypresses
    key_pairs = pair_keys(keys)
    hits = miss = fa = 0
    comm_rts = []
    for p in prompts:
        window = p['t'] + 15
        candidates = [kp for kp in key_pairs if not kp['matched'] and kp['t_press']>=p['t'] and kp['t_press']<=window]
        if p['kind']=='own':
            if candidates:
                best = min(candidates, key=lambda x: x['t_press'])
                best['matched']=True
                hits += 1
                comm_rts.append(best['dur'])
            else:
                miss += 1
        else:
            if candidates:
                best = min(candidates, key=lambda x: x['t_press'])
                best['matched']=True
                fa += 1
    comms_score = hits - fa
    comms_total = len([p for p in prompts if p['kind']=='own'])
    comms_point_accuracy = comms_score / comms_total if comms_total else 0.0
    mean_comms_rt = mean(comm_rts) if comm_rts else 0.0

    return track_point_accuracy, resman_point_accuracy, sysmon_point_accuracy, comms_point_accuracy, mean_sysmon_rt, mean_comms_rt

def main():
    # Set up command-line arguments for input/output files and binning options
    p = argparse.ArgumentParser()
    p.add_argument('input_dir')
    p.add_argument('output_csv')
    p.add_argument('--bin-size', type=int, default=None,
               help='Optional bin size in seconds (e.g., 60)')
    p.add_argument('--bin-overlap', type=float, default=0,
               help='Overlap percentage for bins (e.g., 50 for 50%% overlap)')
    args = p.parse_args()

    # Find all relevant CSV files in the input directory
    files = [
        f for f in glob.glob(os.path.join(args.input_dir, '*.csv'))
        if '_block_' in os.path.basename(f)
           and not os.path.basename(f).endswith('_comms_log.csv')
    ]

    rows = []
    for path in tqdm(files, desc='Scoring', colour='green'):
        # Get participant, session, and condition info from filename and order map
        parts = os.path.splitext(os.path.basename(path))[0].split('_')
        part, ses = parts[0], parts[1]
        block = int(parts[parts.index('block')+1])
        if part not in order_map:
            print(f"Skipping {part} (no order found)")
            continue
        order_str = order_map[part].replace('-', '')
        if block > len(order_str):
            print(f"Skipping {part} block {block} (order too short)")
            continue
        condition = order_str[block - 1]

        # Read all events from the file
        dt0, perf, prompts, keys = read_csv_events(path)

        # Find the latest time in the file to set bin edges
        max_time = max([r['t'] for r in perf] + [p['t'] for p in prompts] + [k['t'] for k in keys], default=0)

        if args.bin_size:
            # If binning is requested, divide the data into overlapping time bins
            step = int(args.bin_size * (1 - args.bin_overlap / 100))
            bin_edges = list(range(0, int(max_time) - args.bin_size + 1, step))
            for b_start in bin_edges:
                b_end = b_start + args.bin_size
                # Only use events that fall within this bin
                b_perf = [r for r in perf if b_start <= r['t'] < b_end]
                b_prompts = [p for p in prompts if b_start <= p['t'] < b_end]
                b_keys = [k for k in keys if b_start <= k['t'] < b_end]

                # Score performance for this bin
                track, resman, sysmon, comms, mean_sysmon_rt, mean_comms_rt = \
                    compute_point_metrics(b_perf, b_prompts, b_keys)

                row = {
                    'participant': part,
                    'session': ses,
                    'condition': condition,
                    'bin_start': b_start,
                    'bin_end': b_end,
                    'track_point_accuracy': track,
                    'resman_point_accuracy': resman,
                    'sysmon_point_accuracy': sysmon,
                    'comms_point_accuracy': comms,
                    'mean_sysmon_response_time': mean_sysmon_rt,
                    'mean_comms_response_time':    mean_comms_rt,
                }
                rows.append(row)
        else:
            # If no binning, score the whole file as one block
            track, resman, sysmon, comms, mean_sysmon_rt, mean_comms_rt = \
                compute_point_metrics(perf, prompts, keys)
            rows.append({
                'participant': part,
                'session': ses,
                'condition': condition,
                'track_point_accuracy': track,
                'resman_point_accuracy': resman,
                'sysmon_point_accuracy': sysmon,
                'comms_point_accuracy': comms,
                'mean_sysmon_response_time': mean_sysmon_rt,
                'mean_comms_response_time':    mean_comms_rt,
            })

    # Write all results to a CSV file
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'participant', 'session', 'condition', 'bin_start', 'bin_end',
            'track_point_accuracy', 'resman_point_accuracy',
            'sysmon_point_accuracy', 'comms_point_accuracy', 'mean_sysmon_response_time', 'mean_comms_response_time'
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows to {args.output_csv}")

if __name__ == '__main__':
    # Run the main function if this file is executed directly
    main()

# Example usage:
# python matb_point_accuracy.py ./input_dir output_point_accuracy.csv --bin-size 60 --bin-