from rqa.utils import norm_utils, plot_utils, output_io_utils
from rqa.utils import rqa_utils_cpp
import os

def crossRQA(data1, data2, params):
    """
    Perform Cross Recurrence Quantification Analysis (RQA).
    Parameters:
        data1 (np.ndarray): Time series data 1.
        data2 (np.ndarray): Time series data 2.
        params (dict): Dictionary of RQA parameters:
                       - norm, eDim, tLag, rescaleNorm, radius, tmin, minl,
                         doPlots, plotMode, phaseSpace, doStatsFile
    """

    # Normalize data
    dataX1 = norm_utils.normalize_data(data1, params['norm'])
    dataX2 = norm_utils.normalize_data(data2, params['norm'])

    # Compute distance maxtrix for RQA
    ds = rqa_utils_cpp.rqa_dist(dataX1, dataX2, dim=params['eDim'], lag=params['tLag'])

    # Perform CRQA calculations
    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(ds["d"], rescale=params['rescaleNorm'], rad=params['radius'], diag_ignore=0, minl=params['minl'], rqa_mode="cross")

    # Print stats
    if err_code == 0:
        if params['showMetrics']:
            print(f"%REC: {float(rs['perc_recur']):.3f} | %DET: {float(rs['perc_determ']):.3f} | MaxLine: {float(rs['maxl_found']):.2f}")
            print(f"Mean Line Length: {float(rs['mean_line_length']):.2f} | SD Line Length: {float(rs['std_line_length']):.2f} | Line Count: {float(rs['count_line']):.2f}")
            print(f"ENTR: {float(rs['entropy']):.3f} | LAM: {float(rs['laminarity']):.3f} | TT: {float(rs['trapping_time']):.3f}")
            print(f"Vmax: {float(rs['vmax']):.2f} | Divergence: {float(rs['divergence']):.3f}") 
            print(f"Trend_Lower: {float(rs['trend_lower_diag']):.3f} | Trend_Upper {float(rs['trend_upper_diag']):.3f}")
    else:
        print("Error in RQA computation. Check parameters and data.")

    # Plot results
    # plotMode: 'none', 'rp', 'rp_timeseries',
    # Plot results
    if 'rp' in params['plotMode']:
        save_path = None
        if params.get('saveFig', False):
            # prefer an explicit savePath if provided
            save_path = params.get('savePath', None)
            if save_path is None:
                # fallback to old behavior or saveDir/saveName if you want
                save_dir = params.get('saveDir', os.path.join('images', 'rqa'))
                save_name = params.get('saveName', 'crossRQA_plot.svg')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_name)

        plot_utils.plot_rqa_results(
            dataX=dataX1,
            dataY=dataX2,
            td=td,
            plot_mode=params['plotMode'],
            point_size=params.get('pointSize', 3),
            save_path=save_path
        )

    # Write stats
    if params['doStatsFile']:
        output_io_utils.write_rqa_stats("CrossRQA", params, rs, err_code)