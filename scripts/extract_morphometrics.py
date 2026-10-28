import sys, os
import argparse
import glob
import numpy as np
import nibabel as nib
from scipy.interpolate import interp1d
import pandas as pd
import spinalcordtoolbox.utils as sct
from spinalcordtoolbox.scripts import sct_process_segmentation

"""
This script computes spinal cord morphometrics (e.g., CSA) from T2-weighted data.

The script :
1. Computes morphometrics using `sct_process_segmentation` per slice.
2. Gets the z-slices corresponding to each disc label.
3. Using the PMJ distances from the `sct_process_segmentation` output, it adds the PMJ distances for each disc label. 
4. Interpolates PMJ distances for each Vertebral Level, and computes morphometrics at each interpolated PMJ distance 
   (at 0.1 mm intervals between each Vertebral Level) using `sct_process_segmentation`

Outputs :
- {subject}_per_slice.csv : CSV files containing morphometrics per slice
- {subject}_PMJ_dist.csv : CSV files with Slice indexes and PMJ distances for each disc label
- {subject}_PMJ_dist_interp.csv : CSV files with the interpolated PMJ distances
- {subject}_temp_pmj_{level}.csv: Temporary CSV files for each interpolated PMJ distance
- {subject}_interpolated_morphometrics.csv : A combined CSV containing CSA values for each interpolated distance. This final CSV also
   contains age and sex information, taken from the `participants.tsv` file.

Usage : 
    The script can be run with `sct_run_batch` using the wrapper script `wrapper_rootlets.sh` as follows:
        
        sct_run_batch -config config/config.yaml -script wrappers/wrapper_extract_morphometrics.sh

Author: Samuelle St-Onge

"""

def run_sct_process_segmentation_per_slice(pmj, t2w_seg_file, output_per_slice_csv):

    """
    This function computes sct_process_segmentation to get the PMJ distances of each slice

    Args:
        pmj: Path to the PMJ label file
        t2w_seg_file: Path to the T2w SC segmentation file
        output_per_slice_csv: the output CSV file containing morphometrics per slice

    Output:
        {subject}_per_slice.csv : CSV files containing morphometrics per slice

    """

    sct_process_segmentation.main([
        '-i', t2w_seg_file,
        '-pmj', pmj,
        '-perslice', '1',
        '-o', output_per_slice_csv,
        '-append', '1'
    ])

def get_disc_label_PMJ_dist(subject, per_slice_csv, output_csv_dir, label_file, output_PMJ_dist_csv):
    """
    Get the PMJ distances of each disc label

    This function:
    1. Generates a CSV file containing the slice indexes corresponding to each disc label ({subject}_PMJ_dist.csv)
    2. Using a previously generated CSV file from sct_process_segmentation per slice, gets the PMJ distances of the slice indexes of each disc label
       and adds it to the {subject}_PMJ_dist.csv.
    
    Args:
        subject: participant ID 
        per_slice_csv: CSV file containing morphometrics per slice 
        output_csv_dir: path to output csv files
        label_file: file containing disc labels
        output_PMJ_dist_csv : the output CSV file containing the list of disc labels with their corresponding slice indexes and PMJ distances

    Output:
        {subject}_PMJ_dist.csv : CSV file containing list of disc labels with their corresponding slice indexes and PMJ distances

    This function was inspired by : https://github.com/sct-pipeline/pmj-based-csa/blob/main/get_disc_slice.py 
    """

    # Read label file
    labels = nib.load(label_file)
    
    # Get labels and their respective z-slice indexes
    z_coords = np.where(labels.get_fdata() != 0)[-1] # Get the z coordinates of the 
    label_values = labels.get_fdata()[np.where(labels.get_fdata() != 0)]
    z = []
    i = 0
    log = pd.DataFrame(columns=['Subject', 'Level', 'Slices'])
    for z in z_coords:
        log = log.append({'Subject': subject, 'Level': label_values[i], 'Slices': z}, ignore_index=True)
        i = i + 1 
    log = log.sort_values(by="Level").reset_index(drop=True)
    log.to_csv(os.path.join(output_csv_dir, f"{subject}_PMJ_dist.csv"), index=False)

    # Merge the PMJ distances from the per_per_slice_csv 
    per_slice_df = pd.read_csv(per_slice_csv) 
    slice_col = "Slice (I->S)"
    pmj_col = "DistancePMJ"   

    log = log.merge(
        per_slice_df[[slice_col, pmj_col]],
        left_on="Slices",
        right_on=slice_col,
        how="left"
    )

    # Save
    log.to_csv(output_PMJ_dist_csv, index=False)


def compute_interpolated_morphometrics(
    subject,
    output_csv_path,
    PMJ_distances_csv,
    pmj,
    t2w_seg_file,
    participants_info,
    interp_PMJ_dist_csv,
    final_csv_filename,
    max_level=21.0
):
    """
    Incremental computation of interpolated morphometrics:
    Only compute missing VertLevels up to max_level.
    Append results to the existing final CSV.

    Args:
        subject: participant ID
        output_csv_path: folder containing CSV files
        PMJ_distances_csv: CSV with PMJ distances per level
        pmj: PMJ label file path
        t2w_seg_file: T2w SC segmentation file path
        participants_info: participants.tsv path
        interp_PMJ_dist_csv: intermediate interpolated PMJ CSV
        final_csv_filename: output CSV file for morphometrics
        max_level: maximum vertebral level to compute (default: 21.0)
    """

    # Read existing final CSV if it exists
    if os.path.exists(final_csv_filename):
        final_results_df = pd.read_csv(final_csv_filename)
        computed_levels = final_results_df['VertLevel'].unique()
    else:
        final_results_df = pd.DataFrame()
        computed_levels = []

    # Read PMJ distances and generate levels
    pmj_df = pd.read_csv(PMJ_distances_csv)
    if 'DistancePMJ' not in pmj_df.columns:
        raise ValueError(f"Missing 'DistancePMJ' column in {PMJ_distances_csv}")

    levels = pmj_df['Level'].values
    pmj_distances = pmj_df['DistancePMJ'].values
    interp_func = interp1d(levels, pmj_distances, kind='linear', fill_value='extrapolate')

    # Generate all levels up to max_level
    all_levels = np.arange(int(levels.min()), max_level + 0.1, 0.1)
    all_levels = np.round(all_levels, 1)

    # Only keep levels not already computed
    levels_to_compute = [lvl for lvl in all_levels if lvl not in computed_levels]
    if not levels_to_compute:
        print(f"All levels up to {max_level} already computed for {subject}. Nothing to do.")
        return

    # Compute interpolated PMJ distances for missing levels
    interp_distances = interp_func(levels_to_compute)

    # Read participant info
    df_participants_info = pd.read_csv(participants_info, sep='\t').rename(columns={'participant_id': 'subject'})
    
    # List to store new results
    new_results = []

    for level, distance_pmj in zip(levels_to_compute, interp_distances):
        temp_csv_filename = os.path.join(output_csv_path, f"{subject}_temp_pmj_{level}.csv")

        # Skip if temp file already exists (safety)
        if os.path.exists(temp_csv_filename):
            print(f"Temporary file exists: {temp_csv_filename}, skipping level {level}")
            continue

        # Run sct_process_segmentation
        sct_process_segmentation.main([
            '-i', t2w_seg_file,
            '-pmj', pmj,
            '-pmj-distance', str(distance_pmj),
            '-pmj-extent', '3',
            '-perlevel', '1',
            '-o', temp_csv_filename,
        ])

        # Read temp results
        temp_df = pd.read_csv(temp_csv_filename)
        temp_df['VertLevel'] = level
        temp_df['DistancePMJ'] = distance_pmj
        temp_df['subject'] = subject

        # Rename columns
        temp_df = temp_df.rename(columns={
            'MEAN(area)': 'CSA',
            'MEAN(diameter_AP)': 'AP_diameter',
            'MEAN(diameter_RL)': 'RL_diameter',
            'MEAN(eccentricity)': 'eccentricity',
            'MEAN(solidity)': 'solidity'
        })

        # Keep only relevant columns
        temp_df = temp_df[['subject', 'VertLevel', 'DistancePMJ', 'CSA', 'AP_diameter', 'RL_diameter', 'eccentricity', 'solidity']]

        new_results.append(temp_df)

        print(f"Processed VertLevel {level} (PMJ distance: {distance_pmj})")

    # Concatenate new results to existing CSV
    if new_results:
        new_results_df = pd.concat(new_results, ignore_index=True)
        new_results_df = new_results_df.merge(df_participants_info[['subject', 'age', 'sex']], on='subject', how='left')

        if final_results_df.empty:
            final_results_df = new_results_df
        else:
            final_results_df = pd.concat([final_results_df, new_results_df], ignore_index=True)
            final_results_df = final_results_df.sort_values('VertLevel').reset_index(drop=True)

        # Save updated CSV
        final_results_df.to_csv(final_csv_filename, index=False)
        print(f"Updated morphometrics saved to: {final_csv_filename}")
    else:
        print("No new levels were computed.")

def compute_missing_levels(subject, final_csv, output_csv_dir, pmj, t2w_seg_file, participants_info, max_level=21.0, step=0.1):
    """
    Incrementally compute missing levels up to max_level.
    Fills zeros if PMJ distance is beyond the segmented SC.

    Args:
        subject: str, participant ID
        final_csv: str, path to existing _interpolated_morphometrics.csv
        output_csv_dir: folder to store temporary CSVs
        pmj: path to PMJ label file
        t2w_seg_file: path to T2w SC segmentation
        participants_info: path to participants.tsv
        max_level: float, maximum vertebral level to compute
        step: float, step size for levels (default 0.1)
    """

    import glob

    # Load existing interpolated morphometrics
    if os.path.exists(final_csv):
        df_final = pd.read_csv(final_csv)
    else:
        df_final = pd.DataFrame(columns=['subject', 'VertLevel', 'DistancePMJ', 'CSA', 
                                         'AP_diameter', 'RL_diameter', 'eccentricity', 'solidity'])
    
    # Determine levels already computed
    existing_levels = df_final['VertLevel'].unique() if not df_final.empty else []

    # Generate full set of levels to compute
    all_levels = np.round(np.arange(0.0, max_level + step, step), 1)

    # Only compute missing levels
    levels_to_compute = [lvl for lvl in all_levels if lvl not in existing_levels]

    if not levels_to_compute:
        print(f"All levels up to {max_level} already computed for {subject}.")
        return df_final

    print(f"Computing missing levels for {subject}: {levels_to_compute[0]} -> {levels_to_compute[-1]}")

    # Load participants info
    df_participants_info = pd.read_csv(participants_info, sep='\t').rename(columns={'participant_id': 'subject'})
    
    all_results = []

    # Load per-slice PMJ distances
    perslice_csv = os.path.join(output_csv_dir, f"{subject}_per_slice.csv")
    perslice_df = pd.read_csv(perslice_csv)

    for level in levels_to_compute:
        # Get closest PMJ distance for this level
        try:
            distance_pmj = perslice_df.loc[perslice_df["DistancePMJ"].notna(), "DistancePMJ"].iloc[int(level)]
        except IndexError:
            distance_pmj = None

        temp_csv_filename = os.path.join(output_csv_dir, f"{subject}_temp_pmj_{level}.csv")

        if distance_pmj is None or distance_pmj > perslice_df["DistancePMJ"].max():
            # Beyond segmentation → fill zeros
            print(f"Level {level:.1f} beyond segmentation, filling zeros.")
            temp_df = pd.DataFrame({
                'MEAN(area)': [0]
            })
        else:
            # Call sct_process_segmentation
            try:
                sct_process_segmentation.main([
                    '-i', t2w_seg_file,
                    '-pmj', pmj,
                    '-pmj-distance', str(distance_pmj),
                    '-pmj-extent', '3',
                    '-perlevel', '1',
                    '-o', temp_csv_filename,
                ])
                temp_df = pd.read_csv(temp_csv_filename)
            except ValueError:
                # If segmentation does not cover distance, fill zeros
                print(f"Warning: Level {level:.1f} could not be processed, filling zeros.")
                temp_df = pd.DataFrame({
                    'CSA': [0],
                    'AP_diameter': [0],
                    'RL_diameter': [0],
                    'eccentricity': [0],
                    'solidity': [0]
                })

        # Add metadata columns
        temp_df['VertLevel'] = level
        temp_df['DistancePMJ'] = distance_pmj if distance_pmj is not None else np.nan
        temp_df['subject'] = subject

        all_results.append(temp_df[['subject','MEAN(area)',]])

    # Combine old + new
    if all_results:
        new_results_df = pd.concat(all_results, ignore_index=True)
        df_final = pd.concat([df_final, new_results_df], ignore_index=True)
        df_final = df_final.sort_values('VertLevel').reset_index(drop=True)

    # Merge age/sex info
    df_final = df_final.merge(df_participants_info[['subject','age','sex']], on='subject', how='left')

    # Save updated CSV
    df_final.to_csv(final_csv, index=False)
    print(f"Updated _interpolated_morphometrics.csv saved for {subject}.")

    # Cleanup temp files
    temp_files = glob.glob(os.path.join(output_csv_dir, f"{subject}_temp_pmj_*.csv"))
    for f in temp_files:
        os.remove(f)
    
    # Append new results to existing CSV
    if all_results:
        df_new = pd.concat(all_results, ignore_index=True)
        df_final = pd.concat([df_existing, df_new], ignore_index=True).sort_values('VertLevel')
        df_final.to_csv(final_csv, index=False)
        print(f"Updated {final_csv} with missing levels.")


def main(subject, data_path, path_output, subject_dir, file_t2):

    # Define paths
    t2w_seg_file = os.path.join(subject_dir, f"{file_t2}_label-SC_seg.nii.gz")
    t2w_disc_labels = os.path.join(subject_dir, f"{file_t2}_labels-disc-manual.nii.gz")
    t2w_pmj_label = os.path.join(subject_dir, f"{file_t2}_label-PMJ_dlabel.nii.gz")
    participants_info = os.path.join(data_path, 'participants.tsv')

    # Define output CSV files
    output_csv_dir = os.path.join("results/morphometrics")
    os.makedirs(output_csv_dir, exist_ok=True) # Create a folder named "morphometrics" inside the output results folder
    output_per_slice_csv = os.path.join(output_csv_dir, f"{subject}_per_slice.csv")
    output_PMJ_dist_csv = os.path.join(output_csv_dir, f"{subject}_PMJ_dist.csv")
    interp_PMJ_dist_csv = os.path.join(output_csv_dir, f"{subject}_PMJ_dist_interp.csv")
    final_csv = os.path.join(output_csv_dir, f"{subject}_interpolated_morphometrics.csv")

    # Define max vert level
    max_level = 21.0

    # # Step 1 : Run sct_process_segmentation per slice to get the PMJ distances of each slice
    # run_sct_process_segmentation_per_slice(
    #     pmj=t2w_pmj_label,
    #     t2w_seg_file=t2w_seg_file,
    #     output_per_slice_csv=output_per_slice_csv
    #     )
    
    # # Step 2 : Get the disc label slices and add the PMJ distances of each disc label

    # get_disc_label_PMJ_dist(
    #     subject, 
    #     output_per_slice_csv,
    #     output_csv_dir, 
    #     t2w_disc_labels,
    #     output_PMJ_dist_csv=output_PMJ_dist_csv)
        
    # Step 2: Compute missing levels
    final_csv = os.path.join(output_csv_dir, f"{subject}_interpolated_morphometrics.csv")
    compute_missing_levels(
        subject=subject,
        final_csv=final_csv,
        output_csv_dir=output_csv_dir,
        pmj=t2w_pmj_label,
        t2w_seg_file=t2w_seg_file,
        participants_info=participants_info,
        max_level=max_level,
        step=0.1
    )
    
    # Step 3 : Interpolate the PMJ distances and run sct_process_segmentation for all interpolated PMJ distances (up to 21.0)
    # print(f"Computing interpolated morphometrics incrementally up to level {max_level} for subject {subject}")
    # compute_interpolated_morphometrics(
    #     subject=subject,
    #     output_csv_path=output_csv_dir,
    #     PMJ_distances_csv=output_PMJ_dist_csv,
    #     pmj=t2w_pmj_label,
    #     t2w_seg_file=t2w_seg_file,
    #     participants_info=participants_info,
    #     interp_PMJ_dist_csv=interp_PMJ_dist_csv,
    #     final_csv_filename=final_csv,
    #     max_level=max_level
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run morphometric extraction for one subject")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001)")
    parser.add_argument("--data-path", required=True, help="Path to raw data")
    parser.add_argument("--path-output", required=True, help="Path to output results")
    parser.add_argument("--subject-dir", required=True, help="Path to subject folder (e.g., sub-001)")
    parser.add_argument("--file-t2", required=True, help="T2-weighted image prefix (e.g., sub-01_T2w)")

    args = parser.parse_args()

    main(args.subject, args.data_path, args.path_output, args.subject_dir, args.file_t2)
