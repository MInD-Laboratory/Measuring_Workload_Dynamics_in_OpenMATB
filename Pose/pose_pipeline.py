#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose_pipeline.py

Standalone facial pose pipeline 

What this script does (in order), controlled by flags at the top:

  1) Load raw OpenPose CSVs (x1,y1,prob1,...,x70,y70,prob70) from RAW_DIR
  2) Filter to relevant keypoints (your sets)                [RUN_FILTER]
  3) Mask low-confidence (conf < CONF_THRESH) to NaN         [RUN_MASK]
  4) Interpolate short gaps (≤ MAX_INTERP_RUN) + Butterworth [RUN_INTERP_FILTER]
  5) Normalize to screen size (2560×1440)                   [RUN_NORM]
  6) Build templates (global + per-participant)             [RUN_TEMPLATES]
  7) Features:
       A) Procrustes vs global template (windowed 60s, 50% overlap)
       A) Procrustes vs participant template (same)
       B) Original (no Procrustes), same windowing           [RUN_FEATURES_*]
     - Per-metric: drop windows containing any NaNs
     - Save three CSVs: procrustes_global, procrustes_participant, original
  8) Interocular scaling + linear metrics (vel, acc, RMS)    [RUN_LINEAR]
     - Save three CSVs for linear metrics corresponding to step 7 outputs.

A JSON summary is saved with:
  - config & flags,
  - per-file masking stats,
  - windows dropped (total & per metric) per route,
  - template info,
  - any errors encountered.

Assumptions:
  - Filenames are exactly "<participantID>_<condition>.csv" e.g., "472_H.csv".
  - Conditions are L/M/H (you can add more; parser is tolerant).
  - Image dimensions are 2560×1440.
  - Sampling is 60 Hz.

"""

from __future__ import annotations  # enable postponed evaluation of type annotations (Python 3.7+ behavior)
from dataclasses import asdict  # import asdict to serialize dataclass instances to dicts
from pathlib import Path  # Path objects for filesystem paths
import json  # JSON read/write for summary artifacts
import textwrap  # pretty-printing multi-line flag dumps
import sys  # access to sys.exit and argv
import numpy as np  # numerical arrays and vectorized ops
import pandas as pd  # dataframes for CSV I/O and tabular manipulation
from tqdm import tqdm  # progress bars for loops

# Ensure the project root (the folder containing utils/) is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pull config, flags, and capability toggles from utils package
from utils import (
    CFG, Config,  # configuration dataclass instance and type
    RUN_FILTER, RUN_MASK, RUN_INTERP_FILTER, RUN_NORM, RUN_TEMPLATES,  # step toggles
    RUN_FEATURES_PROCRUSTES_GLOBAL, RUN_FEATURES_PROCRUSTES_PARTICIPANT,  # feature routes (Procrustes)
    RUN_FEATURES_ORIGINAL, RUN_LINEAR, SCALE_BY_INTEROCULAR,  # original route + linear metrics + scaling mode
    SAVE_REDUCED, SAVE_MASKED, SAVE_INTERP_FILTERED, SAVE_NORM,  # intermediate save toggles
    SAVE_PER_FRAME_PROCRUSTES_GLOBAL, SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT,  # per-frame outputs
    SAVE_PER_FRAME_ORIGINAL,
    OVERWRITE, OVERWRITE_TEMPLATES, SCIPY_AVAILABLE  # overwrite behavior and SciPy availability
)

# Import IO and utility primitives for pipeline steps
from utils.io_utils import (
    ensure_dirs, load_raw_files, parse_participant_condition,  # directory setup, raw file listing, name parsing
    detect_conf_prefix_case_insensitive, relevant_indices, filter_df_to_relevant,  # column detection and selection
    confidence_mask, write_per_frame_metrics, save_json_summary  # masking, per-frame writer, summary JSON helper
)

# Signal processing primitives (interp + filtering)
from utils.signal_utils import interpolate_run_limited, butterworth_segment_filter
# Normalization helpers (screen scaling + interocular distance series)
from utils.normalize_utils import normalize_to_screen, interocular_series
# Feature calculation helpers (per-frame features and linear-from-perframe consolidation)
from utils.features_utils import (
    procrustes_features_for_file, original_features_for_file,
    compute_linear_from_perframe_dir
)
# Windowing helpers (index generation, window summaries, and linear metrics)
from utils.window_utils import windows_indices, window_features, is_distance_like_metric, linear_metrics


def run_pipeline():  # main entry point that orchestrates all steps
    def want_any_steps_1_to_7() -> bool:  # nested helper: are any preprocessing/feature steps requested?
        return any([
            RUN_FILTER, RUN_MASK, RUN_INTERP_FILTER, RUN_NORM, RUN_TEMPLATES,
            RUN_FEATURES_PROCRUSTES_GLOBAL, RUN_FEATURES_PROCRUSTES_PARTICIPANT, RUN_FEATURES_ORIGINAL
        ])  # True if any of steps 2–7 are enabled

    ensure_dirs()  # make sure output directory tree exists before doing any work

    # ---------------- Linear-only mode --------------------------------------
    if RUN_LINEAR and not want_any_steps_1_to_7():  # if we only want step 8 and have per-frame on disk
        lm_dir = Path(CFG.OUT_BASE) / "linear_metrics"  # path to linear metric output directory
        lm_dir.mkdir(parents=True, exist_ok=True)  # create directory if missing

        def compute_linear_for_csv(out_name: str) -> dict:  # inner helper to process one route's per-frame dir
            src = (  # infer source route from the desired output filename
                "procrustes_global" if "procrustes_global" in out_name else
                "procrustes_participant" if "procrustes_participant" in out_name else
                "original"
            )
            per_frame_dir = Path(CFG.OUT_BASE) / "features" / "per_frame" / src  # locate per-frame CSVs for route
            if not per_frame_dir.exists():  # if per-frame data missing, we can't compute linear metrics
                print(f"[skip] No per-frame dir for source '{src}': {per_frame_dir}")  # notify and skip
                return {}  # return empty drop stats
            out_path = lm_dir / out_name  # compute output path for aggregated linear metrics
            drops = compute_linear_from_perframe_dir(  # aggregate velocity/acc/RMS per window across all files
                per_frame_dir, out_path, CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )
            print(f"[OK] Wrote {out_path}")  # confirm write
            return drops  # return drop counts per metric

        srcs = []  # will hold the routes we actually process
        for src, flag in [  # iterate over the three feature routes with their flags
            ("procrustes_global", RUN_FEATURES_PROCRUSTES_GLOBAL),
            ("procrustes_participant", RUN_FEATURES_PROCRUSTES_PARTICIPANT),
            ("original", RUN_FEATURES_ORIGINAL),
        ]:
            per_frame_dir = Path(CFG.OUT_BASE) / "features" / "per_frame" / src  # directory for route
            has_files = per_frame_dir.exists() and any(per_frame_dir.glob("*.csv"))  # check if per-frame CSVs exist
            if flag or has_files:  # include route if flag enabled or files already present
                srcs.append(src)  # add route to process list

        if not srcs:  # no routes available → write a minimal summary and exit cleanly
            summary = {
                "config": asdict(CFG),  # snapshot config
                "flags": {  # record all toggles for transparency
                    "RUN_FILTER": RUN_FILTER, "RUN_MASK": RUN_MASK, "RUN_INTERP_FILTER": RUN_INTERP_FILTER,
                    "RUN_NORM": RUN_NORM, "RUN_TEMPLATES": RUN_TEMPLATES,
                    "RUN_FEATURES_PROCRUSTES_GLOBAL": RUN_FEATURES_PROCRUSTES_GLOBAL,
                    "RUN_FEATURES_PROCRUSTES_PARTICIPANT": RUN_FEATURES_PROCRUSTES_PARTICIPANT,
                    "RUN_FEATURES_ORIGINAL": RUN_FEATURES_ORIGINAL, "RUN_LINEAR": RUN_LINEAR,
                    "SAVE_REDUCED": SAVE_REDUCED, "SAVE_MASKED": SAVE_MASKED,
                    "SAVE_INTERP_FILTERED": SAVE_INTERP_FILTERED, "SAVE_NORM": SAVE_NORM,
                    "OVERWRITE": OVERWRITE, "OVERWRITE_TEMPLATES": OVERWRITE_TEMPLATES,
                    "SCALE_BY_INTEROCULAR": SCALE_BY_INTEROCULAR
                },
                "masking_overall": {},  # no masking in linear-only mode
                "window_drops": {  # no window drops if we didn't compute anything
                    "procrustes_global": {}, "procrustes_participant": {}, "original": {}, "linear_metrics": {}
                }
            }
            save_json_summary(Path(CFG.OUT_BASE) / "pipeline_summary.json", summary)  # persist summary JSON
            print("No per-frame feature CSVs found. Run Step 7 first or enable a RUN_FEATURES_* flag.")  # guidance
            return  # exit run_pipeline

        linear_drop_totals = {}  # store per-route drop statistics
        if "procrustes_global" in srcs:  # process global-template route if requested
            linear_drop_totals["procrustes_global"] = compute_linear_for_csv("procrustes_global_linear.csv")
        if "procrustes_participant" in srcs:  # process participant-template route if requested
            linear_drop_totals["procrustes_participant"] = compute_linear_for_csv("procrustes_participant_linear.csv")
        if "original" in srcs:  # process original (no Procrustes) route if requested
            linear_drop_totals["original"] = compute_linear_for_csv("original_linear.csv")

        summary = {  # build final summary JSON for linear-only path
            "config": asdict(CFG),
            "flags": {
                "RUN_FILTER": RUN_FILTER, "RUN_MASK": RUN_MASK, "RUN_INTERP_FILTER": RUN_INTERP_FILTER,
                "RUN_NORM": RUN_NORM, "RUN_TEMPLATES": RUN_TEMPLATES,
                "RUN_FEATURES_PROCRUSTES_GLOBAL": RUN_FEATURES_PROCRUSTES_GLOBAL,
                "RUN_FEATURES_PROCRUSTES_PARTICIPANT": RUN_FEATURES_PROCRUSTES_PARTICIPANT,
                "RUN_FEATURES_ORIGINAL": RUN_FEATURES_ORIGINAL, "RUN_LINEAR": RUN_LINEAR,
                "SAVE_REDUCED": SAVE_REDUCED, "SAVE_MASKED": SAVE_MASKED,
                "SAVE_INTERP_FILTERED": SAVE_INTERP_FILTERED, "SAVE_NORM": SAVE_NORM,
                "OVERWRITE": OVERWRITE, "OVERWRITE_TEMPLATES": OVERWRITE_TEMPLATES,
                "SCALE_BY_INTEROCULAR": SCALE_BY_INTEROCULAR
            },
            "masking_overall": {},  # still none
            "window_drops": {  # include drop counts from linear aggregation
                "procrustes_global": {}, "procrustes_participant": {}, "original": {}, "linear_metrics": linear_drop_totals
            }
        }
        save_json_summary(Path(CFG.OUT_BASE) / "pipeline_summary.json", summary)  # write summary to disk
        return  # linear-only path ends here

    # ---------------- Full mode (Steps 1–7) ----------------------------------
    files = load_raw_files()  # list raw CSV paths from configured RAW_DIR
    perfile_data = {}  # cache intermediate DataFrames keyed by filename
    perfile_meta = {}  # store per-file metadata (participant, condition, conf prefix)
    perfile_mask_stats = {}  # store masking statistics per file

    print("\n=== Steps 1–3: Load → Filter → Mask ===")  # section header for logs
    for fp in tqdm(files, desc="Load/Filter/Mask", unit="file"):  # iterate with progress bar
        df_raw = pd.read_csv(fp)  # read CSV into DataFrame
        conf_prefix = detect_conf_prefix_case_insensitive(list(df_raw.columns))  # detect confidence prefix ('prob','c',...)
        indices = relevant_indices()  # compute landmark indices we care about

        if RUN_FILTER:  # if filtering step enabled
            df_reduced = filter_df_to_relevant(df_raw, conf_prefix, indices)  # keep only relevant x/y/conf triplets
            if SAVE_REDUCED:  # optionally persist reduced CSV for inspection
                out = Path(CFG.OUT_BASE) / "reduced" / (fp.stem + "_reduced.csv")  # target path
                if OVERWRITE or not out.exists():  # respect overwrite flag
                    df_reduced.to_csv(out, index=False)  # write CSV
        else:  # filtering disabled → pipeline cannot proceed safely
            print("Requested filtering step not enabled; set RUN_FILTER=True.")  # tell user what to change
            return  # abort pipeline

        if RUN_MASK:  # if masking step enabled
            df_masked, stats = confidence_mask(df_reduced, conf_prefix, indices, CFG.CONF_THRESH)  # mask low-confidence samples
            perfile_mask_stats[fp.name] = stats["overall"]  # record overall stats for summary
            if SAVE_MASKED:  # optionally save masked CSV
                out = Path(CFG.OUT_BASE) / "masked" / (fp.stem + "_masked.csv")  # target path
                if OVERWRITE or not out.exists():  # honor overwrite policy
                    df_masked.to_csv(out, index=False)  # write CSV
        else:  # masking disabled → downstream steps expect clean inputs
            print("Requested masking step not enabled; set RUN_MASK=True.")  # guidance
            return  # abort

        perfile_data[fp.name] = {"reduced": df_reduced, "masked": df_masked}  # stash intermediates
        pid, cond = parse_participant_condition(fp.name)  # parse participant and condition from filename
        perfile_meta[fp.name] = {"participant": pid, "condition": cond, "conf_prefix": conf_prefix}  # store meta

    print("\n=== Step 4: Interpolate + Filter ===")  # step header
    if RUN_INTERP_FILTER:  # proceed only if enabled
        if not SCIPY_AVAILABLE:  # check SciPy availability for Butterworth filter
            print("scipy is required for RUN_INTERP_FILTER. Install scipy or disable this step.")  # error message
            return  # cannot continue interpolation+filter step without SciPy
        for fp in tqdm(files, desc="Interp/Filter", unit="file"):  # process each file
            name = fp.name  # cache filename key
            dfm = perfile_data[name]["masked"].copy()  # start from masked data
            for col in dfm.columns:  # iterate through columns
                lc = col.lower()  # lowercase for prefix check
                if lc.startswith("x") or lc.startswith("y"):  # only process coordinate columns
                    dfm[col] = interpolate_run_limited(dfm[col], CFG.MAX_INTERP_RUN)  # interpolate short NaN runs only
                    dfm[col] = butterworth_segment_filter(dfm[col], CFG.FILTER_ORDER, CFG.CUTOFF_HZ, CFG.FPS)  # low-pass filter per contiguous non-NaN segment
            if SAVE_INTERP_FILTERED:  # optionally persist cleaned coordinates
                out = Path(CFG.OUT_BASE) / "interp_filtered" / (fp.stem + "_interp_filt.csv")  # output path
                if OVERWRITE or not out.exists():  # respect overwrite policy
                    dfm.to_csv(out, index=False)  # write CSV
            perfile_data[name]["interp_filt"] = dfm  # store result for subsequent steps
    else:  # step disabled
        print("Requested RUN_INTERP_FILTER=False. Downstream steps may fail.")  # warning
        return  # abort to avoid undefined behavior later

    print("\n=== Step 5: Normalize to screen ===")  # header
    if RUN_NORM:  # only if enabled
        for fp in tqdm(files, desc="Normalize", unit="file"):  # loop files
            name = fp.name  # key
            dfc = perfile_data[name]["interp_filt"]  # take cleaned coords
            df_norm = normalize_to_screen(dfc, CFG.IMG_WIDTH, CFG.IMG_HEIGHT)  # divide x by width, y by height
            if SAVE_NORM:  # optionally save normalized CSV
                out = Path(CFG.OUT_BASE) / "norm_screen" / (fp.stem + "_norm.csv")  # path
                if OVERWRITE or not out.exists():  # overwrite check
                    df_norm.to_csv(out, index=False)  # write CSV
            perfile_data[name]["norm"] = df_norm  # store normalized data
    else:  # normalization off
        print("Requested RUN_NORM=False. Templates and features require normalized coords.")  # warning
        return  # abort because templates/features assume normalized units

    print("\n=== Step 6: Templates (global + per-participant) ===")  # header
    templ_dir = Path(CFG.OUT_BASE) / "templates"  # templates directory
    global_templ_path = templ_dir / "global_template.csv"  # global template CSV path

    part_to_files = {}  # map participant -> list of filenames
    for fp in files:  # group by participant
        pid = perfile_meta[fp.name]["participant"]  # extract participant id
        part_to_files.setdefault(pid, []).append(fp.name)  # append filename under participant key

    def compute_template_across_files(file_names: list[str]) -> pd.DataFrame:  # average template builder
        if not file_names:  # empty list guard
            return pd.DataFrame()  # return empty DataFrame
        cols = perfile_data[file_names[0]]["norm"].columns  # reference column order from first file
        accum = [perfile_data[name]["norm"][cols].astype(float) for name in file_names]  # collect normalized coords
        big = pd.concat(accum, axis=0, ignore_index=True)  # stack all frames across files
        x_cols = [c for c in cols if c.lower().startswith("x")]  # x columns
        y_cols = [c for c in cols if c.lower().startswith("y")]  # y columns
        templ = pd.DataFrame(index=[0], columns=x_cols + y_cols, dtype=float)  # single-row template frame
        templ[x_cols] = big[x_cols].mean(axis=0, skipna=True).values  # mean x per landmark across all frames/files
        templ[y_cols] = big[y_cols].mean(axis=0, skipna=True).values  # mean y per landmark across all frames/files
        return templ  # return the template row

    if RUN_TEMPLATES:  # proceed if templates are requested
        if global_templ_path.exists() and not OVERWRITE_TEMPLATES:  # reuse existing global template when allowed
            global_template = pd.read_csv(global_templ_path)  # load from disk
        else:  # compute new global template
            all_names = [fp.name for fp in files]  # list of all filenames
            global_template = compute_template_across_files(all_names)  # build global template by averaging
            global_template.to_csv(global_templ_path, index=False)  # persist global template

        participant_templates = {}  # per-participant template cache
        for pid, names in tqdm(part_to_files.items(), desc="Participant templates", unit="participant"):  # for each participant
            part_path = templ_dir / f"participant_{pid}_template.csv"  # path to participant-specific template
            if part_path.exists() and not OVERWRITE_TEMPLATES:  # reuse if allowed
                participant_templates[pid] = pd.read_csv(part_path)  # load
            else:  # (re)compute participant template
                templ = compute_template_across_files(names)  # average over that participant's files
                templ.to_csv(part_path, index=False)  # persist
                participant_templates[pid] = templ  # cache
    else:  # templates disabled
        print("RUN_TEMPLATES=False requested. Procrustes features require templates.")  # warning
        return  # abort because Procrustes routes depend on templates

    # ----------------------- Step 7: Features --------------------------------
    print("\n=== Step 7: Features (windowed) ===")  # header
    win = CFG.WINDOW_SECONDS * CFG.FPS  # samples per window (e.g., 60s * 60Hz = 3600 frames)
    hop = int(win * (1.0 - CFG.WINDOW_OVERLAP)) or max(1, win // 2)  # window hop (ensure >=1)
    rel_idxs = relevant_indices()  # recompute relevant landmark indices (used by feature extractors)
    feat_dir = Path(CFG.OUT_BASE) / "features"  # features output directory
    feat_dir.mkdir(parents=True, exist_ok=True)  # ensure it exists

    procrustes_global_rows = []  # row buffers for aggregated window features (global template)
    procrustes_part_rows = []  # row buffers for aggregated window features (participant template)
    procrustes_global_drops_agg = {}  # window drop counters per metric for global route
    procrustes_part_drops_agg = {}  # window drop counters per metric for participant route
    original_rows = []  # row buffers for original route
    original_drops_agg = {}  # window drop counters per metric for original route

    if RUN_FEATURES_PROCRUSTES_GLOBAL or RUN_FEATURES_PROCRUSTES_PARTICIPANT:  # if any Procrustes route enabled
        print("Computing Procrustes features...")  # log
        for fp in tqdm(files, desc="Procrustes features", unit="file"):  # iterate files
            name = fp.name  # key
            pid = perfile_meta[name]["participant"]  # participant id
            cond = perfile_meta[name]["condition"]  # condition label
            df_norm = perfile_data[name]["norm"]  # normalized coordinates for this file

            if RUN_FEATURES_PROCRUSTES_GLOBAL:  # global-template alignment path
                feats = procrustes_features_for_file(df_norm, global_template, rel_idxs)  # per-frame features
                io = interocular_series(df_norm, perfile_meta[name]["conf_prefix"]).values  # per-frame interocular distances
                n_frames = len(io)  # number of frames in the sequence
                if SAVE_PER_FRAME_PROCRUSTES_GLOBAL:  # optionally persist per-frame features
                    write_per_frame_metrics(feat_dir, "procrustes_global", pid, cond, feats, io, n_frames)  # write CSV
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)  # aggregate per-window means + drop stats
                dfw.insert(0, "condition", cond); dfw.insert(0, "participant", pid); dfw.insert(0, "source", "procrustes_global")  # annotate
                procrustes_global_rows.append(dfw)  # buffer
                for k, v in drops.items():  # accumulate drop counts per metric across files
                    procrustes_global_drops_agg[k] = procrustes_global_drops_agg.get(k, 0) + v  # sum

            if RUN_FEATURES_PROCRUSTES_PARTICIPANT:  # participant-template alignment path
                templ = participant_templates[pid]  # select participant-specific template
                feats = procrustes_features_for_file(df_norm, templ, rel_idxs)  # per-frame features
                io = interocular_series(df_norm, perfile_meta[name]["conf_prefix"]).values  # interocular distances
                n_frames = len(io)  # frame count
                if SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT:  # optionally write per-frame CSV
                    write_per_frame_metrics(feat_dir, "procrustes_participant", pid, cond, feats, io, n_frames)  # save
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)  # window aggregation
                dfw.insert(0, "condition", cond); dfw.insert(0, "participant", pid); dfw.insert(0, "source", "procrustes_participant")  # annotate
                procrustes_part_rows.append(dfw)  # buffer
                for k, v in drops.items():  # accumulate drop counts
                    procrustes_part_drops_agg[k] = procrustes_part_drops_agg.get(k, 0) + v  # sum

    if RUN_FEATURES_ORIGINAL:  # original (no alignment) route
        print("Computing Original (no Procrustes) features...")  # log
        for fp in tqdm(files, desc="Original features", unit="file"):  # loop files
            name = fp.name  # key
            pid = perfile_meta[name]["participant"]  # participant id
            cond = perfile_meta[name]["condition"]  # condition label
            df_norm = perfile_data[name]["norm"]  # normalized coordinates

            feats = original_features_for_file(df_norm)  # compute per-frame features directly in normalized coords
            io = interocular_series(df_norm, perfile_meta[name]["conf_prefix"]).values  # interocular distances
            n_frames = len(io)  # frame count
            if SAVE_PER_FRAME_ORIGINAL:  # optionally save per-frame CSV
                write_per_frame_metrics(feat_dir, "original", pid, cond, feats, io, n_frames)  # persist
            dfw, drops = window_features(feats, io, CFG.FPS, win, hop)  # per-window means and drops
            dfw.insert(0, "condition", cond); dfw.insert(0, "participant", pid); dfw.insert(0, "source", "original")  # annotate
            original_rows.append(dfw)  # buffer
            for k, v in drops.items():  # accumulate drops
                original_drops_agg[k] = original_drops_agg.get(k, 0) + v  # sum

    # Save Step 7 CSVs
    if RUN_FEATURES_PROCRUSTES_GLOBAL and procrustes_global_rows:  # if we computed global route features
        pd.concat(procrustes_global_rows, ignore_index=True).to_csv(feat_dir / "procrustes_global_features.csv", index=False)  # write combined CSV
    if RUN_FEATURES_PROCRUSTES_PARTICIPANT and procrustes_part_rows:  # if we computed participant route features
        pd.concat(procrustes_part_rows, ignore_index=True).to_csv(feat_dir / "procrustes_participant_features.csv", index=False)  # write combined CSV
    if RUN_FEATURES_ORIGINAL and original_rows:  # if we computed original route
        pd.concat(original_rows, ignore_index=True).to_csv(feat_dir / "original_features.csv", index=False)  # write combined CSV

    # ----------------------- Step 8: Linear metrics --------------------------
    print("\n=== Step 8: Interocular scaling + linear metrics ===")  # header
    lm_dir = Path(CFG.OUT_BASE) / "linear_metrics"  # linear metrics output directory
    lm_dir.mkdir(parents=True, exist_ok=True)  # ensure exists

    linear_drop_totals = {}  # dictionary to collect drop counts per route
    if RUN_LINEAR:  # only compute if enabled
        if RUN_FEATURES_PROCRUSTES_GLOBAL:  # consume per-frame (global) to produce linear metrics
            linear_drop_totals["procrustes_global"] = compute_linear_from_perframe_dir(
                feat_dir / "per_frame" / "procrustes_global",
                lm_dir / "procrustes_global_linear.csv",
                CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )
        if RUN_FEATURES_PROCRUSTES_PARTICIPANT:  # consume per-frame (participant) to produce linear metrics
            linear_drop_totals["procrustes_participant"] = compute_linear_from_perframe_dir(
                feat_dir / "per_frame" / "procrustes_participant",
                lm_dir / "procrustes_participant_linear.csv",
                CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )
        if RUN_FEATURES_ORIGINAL:  # consume per-frame (original) to produce linear metrics
            linear_drop_totals["original"] = compute_linear_from_perframe_dir(
                feat_dir / "per_frame" / "original",
                lm_dir / "original_linear.csv",
                CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )

    # ----------------------- Summary JSON ------------------------------------
    summary = {  # build final run summary artifact
        "config": asdict(CFG),  # serialize config dataclass to a plain dict
        "flags": {  # record all flags for reproducibility
            "RUN_FILTER": RUN_FILTER, "RUN_MASK": RUN_MASK, "RUN_INTERP_FILTER": RUN_INTERP_FILTER,
            "RUN_NORM": RUN_NORM, "RUN_TEMPLATES": RUN_TEMPLATES,
            "RUN_FEATURES_PROCRUSTES_GLOBAL": RUN_FEATURES_PROCRUSTES_GLOBAL,
            "RUN_FEATURES_PROCRUSTES_PARTICIPANT": RUN_FEATURES_PROCRUSTES_PARTICIPANT,
            "RUN_FEATURES_ORIGINAL": RUN_FEATURES_ORIGINAL, "RUN_LINEAR": RUN_LINEAR,
            "SAVE_REDUCED": SAVE_REDUCED, "SAVE_MASKED": SAVE_MASKED,
            "SAVE_INTERP_FILTERED": SAVE_INTERP_FILTERED, "SAVE_NORM": SAVE_NORM,
            "OVERWRITE": OVERWRITE, "OVERWRITE_TEMPLATES": OVERWRITE_TEMPLATES,
            "SCALE_BY_INTEROCULAR": SCALE_BY_INTEROCULAR
        },
        "masking_overall": perfile_mask_stats,  # aggregate masking statistics per file
        "window_drops": {  # how many windows were dropped per metric per route
            "procrustes_global": procrustes_global_drops_agg if RUN_FEATURES_PROCRUSTES_GLOBAL else {},
            "procrustes_participant": procrustes_part_drops_agg if RUN_FEATURES_PROCRUSTES_PARTICIPANT else {},
            "original": original_drops_agg if RUN_FEATURES_ORIGINAL else {},
            "linear_metrics": linear_drop_totals if RUN_LINEAR else {}
        }
    }
    with open(Path(CFG.OUT_BASE) / "pipeline_summary.json", "w") as f:  # open summary path for writing
        json.dump(summary, f, indent=2)  # pretty-print JSON for human inspection
    print("\nSummary written to:", Path(CFG.OUT_BASE) / "pipeline_summary.json")  # console notice
    print("Done.")  # end-of-run signal


if __name__ == "__main__":  # allow script to run as a module or standalone
    print("POSE PIPELINE — standalone mode")  # banner
    print("Config:")  # header
    for k, v in asdict(CFG).items():  # iterate config fields
        print(f"  {k}: {v}")  # print key/value pairs
    print("\nFlags:")  # header for flags
    print(textwrap.indent(  # indent the next block for readability
        "\n".join([f"{k}: {globals()[k]}" for k in [  # build lines of flag=value using global variables
            "RUN_FILTER","RUN_MASK","RUN_INTERP_FILTER","RUN_NORM",
            "RUN_TEMPLATES","RUN_FEATURES_PROCRUSTES_GLOBAL","RUN_FEATURES_PROCRUSTES_PARTICIPANT",
            "RUN_FEATURES_ORIGINAL","RUN_LINEAR",
            "SAVE_REDUCED","SAVE_MASKED","SAVE_INTERP_FILTERED","SAVE_NORM",
            "OVERWRITE","OVERWRITE_TEMPLATES","SCALE_BY_INTEROCULAR"
        ]]),
        "  "  # two-space indentation prefix
    ))
    if not SCIPY_AVAILABLE and RUN_INTERP_FILTER:  # guard: can't filter without SciPy
        print("\nERROR: scipy is required for RUN_INTERP_FILTER. Install scipy or set RUN_INTERP_FILTER=False.")  # error text
        sys.exit(1)  # hard exit with non-zero status
    run_pipeline()  # invoke main pipeline function
