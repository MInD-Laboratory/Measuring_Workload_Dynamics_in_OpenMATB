# utils/viz_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Patch
from typing import List, Optional, Tuple

from .preprocessing_utils import relevant_indices, detect_conf_prefix_case_insensitive, lm_triplet_colnames
from .features_utils import procrustes_frame_to_template  # if separated; else import the function wherever it lives
from .nb_utils import find_col

# --- Build Procrustes-aligned coordinates for a slice (and a couple features) ---
def procrustes_transform_series(
    df_norm: pd.DataFrame,
    template_df: pd.DataFrame,
    rel_idxs: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    For each frame in df_norm, align relevant landmarks to template_df (single-row).
    Returns:
      - DataFrame with columns x1..y68 (NaN where unavailable)
      - dict of simple per-frame features: head_rotation_angle, blink_dist, mouth_dist
    """
    if rel_idxs is None:
        rel_idxs = relevant_indices()

    # map columns present
    def _cols(df, axis):
        cols = {}
        for i in rel_idxs:
            c = find_col(df, axis, i)
            if c: cols[i] = c
        return cols

    xcols = _cols(df_norm, "x"); ycols = _cols(df_norm, "y")
    templ_xy = np.column_stack([
        [template_df.iloc[0][find_col(template_df, "x", i)] if find_col(template_df, "x", i) else np.nan for i in rel_idxs],
        [template_df.iloc[0][find_col(template_df, "y", i)] if find_col(template_df, "y", i) else np.nan for i in rel_idxs],
    ])

    n = len(df_norm)
    cols_out = []
    for i in range(1, 69):
        cols_out += [f"x{i}", f"y{i}"]
    data_out = np.full((n, len(cols_out)), np.nan, float)

    head_rotation_angle = np.full(n, np.nan, float)
    blink_dist = np.full(n, np.nan, float)
    mouth_dist = np.full(n, np.nan, float)

    def _idx(i):
        try: return rel_idxs.index(i)
        except ValueError: return -1

    Ltop = [_idx(38), _idx(39)]; Lbot = [_idx(41), _idx(42)]
    Rtop = [_idx(44), _idx(45)]; Rbot = [_idx(47), _idx(48)]
    eye_pair = (_idx(37), _idx(46))
    mouth_pair = (_idx(63), _idx(67))

    def _angle(p1, p2): return float(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
    def _blink_ap(top2, bot2):
        top_mean = top2.mean(axis=0); bot_mean = bot2.mean(axis=0)
        return abs(float(top_mean[1] - bot_mean[1]))
    def _mouth_ap(p63, p67): return float(np.linalg.norm(p67 - p63))

    for t in range(n):
        fx = [pd.to_numeric(df_norm.iloc[t][xcols[i]], errors="coerce") if i in xcols else np.nan for i in rel_idxs]
        fy = [pd.to_numeric(df_norm.iloc[t][ycols[i]], errors="coerce") if i in ycols else np.nan for i in rel_idxs]
        frame_xy = np.column_stack([np.asarray(fx, float), np.asarray(fy, float)])
        avail = np.isfinite(frame_xy).all(axis=1) & np.isfinite(templ_xy).all(axis=1)
        ok, s, tx, ty, R, X = procrustes_frame_to_template(frame_xy, templ_xy, avail)  # imported from utils
        if not ok:
            continue

        # fill transformed pose for that frame
        for i in range(1, 69):
            if i in rel_idxs:
                j = rel_idxs.index(i)
                if np.isfinite(X[j]).all():
                    data_out[t, 2*(i-1)  ] = X[j, 0]
                    data_out[t, 2*(i-1)+1] = X[j, 1]

        # basic features
        i37, i46 = eye_pair
        if i37 >= 0 and i46 >= 0 and np.isfinite(X[i37]).all() and np.isfinite(X[i46]).all():
            head_rotation_angle[t] = _angle(X[i37], X[i46])

        def _safe_pts(idxs):
            pts = [X[k] for k in idxs if k >= 0 and np.isfinite(X[k]).all()]
            return np.vstack(pts) if len(pts) == 2 else None

        vals = []
        ltop, lbot = _safe_pts(Ltop), _safe_pts(Lbot)
        rtop, rbot = _safe_pts(Rtop), _safe_pts(Rbot)
        if ltop is not None and lbot is not None: vals.append(_blink_ap(ltop, lbot))
        if rtop is not None and rbot is not None: vals.append(_blink_ap(rtop, rbot))
        if vals: blink_dist[t] = float(np.mean(vals))

        m63, m67 = mouth_pair
        if m63 >= 0 and m67 >= 0 and np.isfinite(X[m63]).all() and np.isfinite(X[m67]).all():
            mouth_dist[t] = _mouth_ap(X[m63], X[m67])

    df_pose = pd.DataFrame(data_out, columns=cols_out)
    feats = {
        "head_rotation_angle": head_rotation_angle,
        "blink_dist": blink_dist,
        "mouth_dist": mouth_dist,
    }
    return df_pose, feats

# --- Interactive viewer ---
def create_interactive_pose_timeseries_viewer(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    features_to_plot: List[str] = ('blink_dist', 'mouth_dist'),
    landmark_subset: Optional[List[int]] = None,
    figsize: tuple = (16, 10),
    fps: Optional[int] = None,   # computed if None
    pose_sampling_hz: int = 60,
    plot_downsample: int = 2,
    window_seconds: int = 240
):
    WINDOW_FRAMES = int(window_seconds * pose_sampling_hz)
    HALF_WIN = WINDOW_FRAMES // 2
    STEP_FRAMES = max(1, plot_downsample)
    if fps is None:
        fps = max(1, pose_sampling_hz // plot_downsample)
    interval_ms = int(1000 / fps)

    if landmark_subset is None:
        landmark_subset = list(range(1, 69))  # 1..68 inclusive
    avail = [lm for lm in landmark_subset if f'x{lm}' in df_raw.columns and f'y{lm}' in df_raw.columns]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(features_to_plot), 2, width_ratios=[1, 1.5],
                          height_ratios=[1] * len(features_to_plot),
                          hspace=0.3, wspace=0.3)
    ax_pose = fig.add_subplot(gs[:, 0])
    ax_times = []
    for i in range(len(features_to_plot)):
        ax = fig.add_subplot(gs[i, 1]) if i == 0 else fig.add_subplot(gs[i, 1], sharex=ax_times[0])
        ax_times.append(ax)

    ax_slider = plt.axes([0.1, 0.02, 0.65, 0.03])
    ax_play   = plt.axes([0.81, 0.02, 0.08, 0.04])

    n_frames = len(df_raw)
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valfmt='%d')
    play_button = Button(ax_play, 'Play')
    state = {'playing': False, 'current_frame': 0}

    def get_bounds(xs, ys):
        vx, vy = xs[~np.isnan(xs)], ys[~np.isnan(ys)]
        if len(vx) == 0: return -1, 1, -1, 1
        cx, cy = np.mean(vx), np.mean(vy)
        rng = max(np.ptp(vx), np.ptp(vy))
        pad = rng * 0.2
        return (cx - rng/2 - pad, cx + rng/2 + pad,
                cy - rng/2 - pad, cy + rng/2 + pad)

    def update_pose(i):
        ax_pose.clear()
        xs, ys, colors = [], [], []
        for lm in avail:
            xs.append(df_raw.loc[i, f'x{lm}']); ys.append(df_raw.loc[i, f'y{lm}'])
            colors.append('blue' if 37 <= lm <= 48 else
                          'red'  if 49 <= lm <= 68 else
                          'green' if 28 <= lm <= 36 else 'gray')
        xs = np.array(xs); ys = np.array(ys); mask = ~np.isnan(xs) & ~np.isnan(ys)
        ax_pose.scatter(xs[mask], ys[mask], c=np.array(colors)[mask], s=20, alpha=0.8)

        def draw(seq, c):
            pts = []
            for lm in seq:
                if lm in avail:
                    j = avail.index(lm)
                    if mask[j]: pts.append([xs[j], ys[j]])
            if len(pts) > 1:
                pts = np.array(pts); ax_pose.plot(pts[:, 0], pts[:, 1], color=c, alpha=0.6, linewidth=1.5)

        draw([37,38,39,40,41,42,37], 'blue')
        draw([43,44,45,46,47,48,43], 'blue')
        draw([49,50,51,52,53,54,55,56,57,58,59,60,49], 'red')
        draw([28,29,30,31,32,33,34,35,36], 'green')

        x_min, x_max, y_min, y_max = get_bounds(xs, ys)
        ax_pose.set_xlim(x_min, x_max); ax_pose.set_ylim(y_max, y_min)  # flip Y
        ax_pose.set_aspect('equal'); ax_pose.set_title(f'Facial Pose - Frame {i}', fontweight='bold')
        ax_pose.grid(True, alpha=0.3)
        ax_pose.legend(handles=[Patch(facecolor='blue', label='Eyes'),
                                Patch(facecolor='red', label='Mouth'),
                                Patch(facecolor='green', label='Nose'),
                                Patch(facecolor='gray', label='Face')],
                       loc='upper right', fontsize=8)

    def update_times(i):
        start = max(0, i - HALF_WIN); end = min(len(df_features), i + HALF_WIN)
        idx = np.arange(start, end); idx_ds = idx[::plot_downsample]
        for ax, feat in zip(ax_times, features_to_plot):
            ax.clear()
            if feat in df_features.columns:
                y = df_features[feat].values
                ax.plot(idx_ds, y[start:end:plot_downsample], '-', alpha=0.8, linewidth=1)
                ax.axvline(i, linestyle='--', linewidth=2, alpha=0.9)
                if i < len(df_features) and np.isfinite(y[i]):
                    ax.scatter(i, y[i], s=80, zorder=5)
                    ax.text(i, y[i], f'{y[i]:.3f}', ha='center', va='bottom',
                            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                ax.set_xlim(start, end)
                ax.set_ylabel(feat, fontweight='bold'); ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Feature "{feat}" not found', transform=ax.transAxes, ha='center', va='center')
        ax_times[-1].set_xlabel('Frame Number', fontweight='bold')

    def update_all(i, *, from_slider=False):
        i = int(np.clip(i, 0, n_frames - 1))
        update_pose(i); update_times(i); state['current_frame'] = i
        if not from_slider and slider.val != i:
            slider.eventson = False; slider.set_val(i); slider.eventson = True
        fig.canvas.draw_idle()

    def on_slider(val):
        if not state['playing']: update_all(int(val), from_slider=True)

    def toggle(_):
        state['playing'] = not state['playing']
        play_button.label.set_text('Pause' if state['playing'] else 'Play')
        (fig._timer.start() if state['playing'] else fig._timer.stop())

    def step():
        if not state['playing']: return
        update_all((state['current_frame'] + max(1, plot_downsample)) % n_frames, from_slider=False)

    update_all(0, from_slider=True)
    slider.on_changed(on_slider); play_button.on_clicked(toggle)
    timer = fig.canvas.new_timer(interval=interval_ms); timer.add_callback(step)
    fig._widgets = {'slider': slider, 'play_button': play_button}; fig._timer = timer

    fig.text(0.02, 0.98,
             f'Instructions:\n• Plot decimation ×{plot_downsample}\n'
             f'• Showing {window_seconds}s window (~{WINDOW_FRAMES} frames @ {pose_sampling_hz} Hz)\n'
             f'• Slider = full-res; Play = +{max(1, plot_downsample)} frame(s)/tick @ ~{fps} Hz',
             transform=fig.transFigure, va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    plt.show()
    return fig
