import os
import numpy as np
import pandas as pd
from tqdm import tqdm

CONFIDENCE_THRESHOLD = 0.5  # If a measurement's confidence is below this, it will be fixed by interpolation

def load_csv(file_path):
    # Load a CSV file into a table
    return pd.read_csv(file_path)

def compute_averages(df):
    # Calculate average positions and confidence for different face parts (center, eyes, pupils)
    averaged_df = pd.DataFrame({
        'center_face_x': df[[f'x{i}' for i in range(27, 36)]].mean(axis=1),
        'center_face_y': df[[f'y{i}' for i in range(27, 36)]].mean(axis=1),
        'center_face_prob': df[[f'prob{i}' for i in range(27, 36)]].mean(axis=1),

        'left_eye_x': df[[f'x{i}' for i in range(36, 42)]].mean(axis=1),
        'left_eye_y': df[[f'y{i}' for i in range(36, 42)]].mean(axis=1),
        'left_eye_prob': df[[f'prob{i}' for i in range(36, 42)]].mean(axis=1),

        'right_eye_x': df[[f'x{i}' for i in range(42, 48)]].mean(axis=1),
        'right_eye_y': df[[f'y{i}' for i in range(42, 48)]].mean(axis=1),
        'right_eye_prob': df[[f'prob{i}' for i in range(42, 48)]].mean(axis=1),

        'left_pupil_x': df['x68'],
        'left_pupil_y': df['y68'],
        'left_pupil_prob': df['prob68'],

        'right_pupil_x': df['x69'],
        'right_pupil_y': df['y69'],
        'right_pupil_prob': df['prob69'],
    })

    # For each face part, if the confidence is low, mark the position as missing
    for part in ['center_face', 'left_eye', 'right_eye', 'left_pupil', 'right_pupil']:
        prob_col = f'{part}_prob'
        for axis in ['x', 'y']:
            val_col = f'{part}_{axis}'
            averaged_df.loc[averaged_df[prob_col] < CONFIDENCE_THRESHOLD, val_col] = np.nan

    # Fill in missing values by guessing (interpolating) between good values
    averaged_df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    return averaged_df

def compute_magnitude(df):
    # For each x/y pair, calculate the overall distance from the origin (magnitude)
    for col in df.columns:
        if col.endswith('_x'):
            y_col = col.replace('_x', '_y')
            if y_col in df.columns:
                mag_col = col.replace('_x', '_magnitude')
                df[mag_col] = np.sqrt(df[col]**2 + df[y_col]**2)
    return df

def compute_combined_eyes(df):
    # Combine left and right pupil positions into an average position and magnitude
    df['avg_pupil_x'] = (df['left_pupil_x'] + df['right_pupil_x']) / 2
    df['avg_pupil_y'] = (df['left_pupil_y'] + df['right_pupil_y']) / 2
    df['avg_pupil_magnitude'] = np.sqrt(df['avg_pupil_x']**2 + df['avg_pupil_y']**2)
    return df

def compute_blink(df_raw):
    # Calculate the blink distance (how open the eyes are) using specific eye landmarks
    top_right_x = df_raw[['x37', 'x38']].mean(axis=1)
    top_right_y = df_raw[['y37', 'y38']].mean(axis=1)
    bottom_right_x = df_raw[['x40', 'x41']].mean(axis=1)
    bottom_right_y = df_raw[['y40', 'y41']].mean(axis=1)
    right_eye_dist = np.sqrt((top_right_x - bottom_right_x)**2 + (top_right_y - bottom_right_y)**2)
    top_left_x = df_raw[['x43', 'x44']].mean(axis=1)
    top_left_y = df_raw[['y43', 'y44']].mean(axis=1)
    bottom_left_x = df_raw[['x46', 'x47']].mean(axis=1)
    bottom_left_y = df_raw[['y46', 'y47']].mean(axis=1)
    left_eye_dist = np.sqrt((top_left_x - bottom_left_x)**2 + (top_left_y - bottom_left_y)**2)

    blink_dist = (right_eye_dist + left_eye_dist) / 2.0
    return blink_dist

def compute_head_rotation(df_raw):
    # Calculate the angle of the head using two points on the face
    dx = df_raw['x45'] - df_raw['x36']
    dy = df_raw['y45'] - df_raw['y36']
    return np.arctan2(dy, dx)

def compute_mouth_distance(df_raw):
    # Calculate how open the mouth is using two mouth landmarks
    return np.sqrt((df_raw['x62'] - df_raw['x66'])**2 + (df_raw['y62'] - df_raw['y66'])**2)

def process_data(input_dir, output_dir):
    # Process all CSV files in the input folder and save results to the output folder
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        file_path = os.path.join(input_dir, csv_file)
        try:
            # Load the raw data
            df_raw = load_csv(file_path)
            # Compute averages and derived features
            averaged_df = compute_averages(df_raw)
            averaged_df = compute_magnitude(averaged_df)
            averaged_df = compute_combined_eyes(averaged_df)
            averaged_df['blink_dist'] = compute_blink(df_raw)
            averaged_df['head_rotation_angle'] = compute_head_rotation(df_raw)
            averaged_df['mouth_dist'] = compute_mouth_distance(df_raw)

            # Save the processed data to a new CSV file
            out_file = os.path.splitext(csv_file)[0] + ".csv"
            out_path = os.path.join(output_dir, out_file)
            averaged_df.to_csv(out_path, index=False)
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

if __name__ == "__main__":
    # Set where to find the raw data and where to save the processed data
    input_directory = 'data/pose/experimental_pose'
    output_directory = 'data/preprocessed_pose/experimental_pose'
    process_data(input_directory,output_directory)