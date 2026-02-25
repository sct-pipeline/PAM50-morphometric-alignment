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
        log = log.append({'Subject': subject, 'Level': 
        label_values[i], 'Slices': z}, ignore_index=True)
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


def compute_interpolated_morphometrics(subject, output_csv_path, PMJ_distances_csv, pmj, t2w_seg_file, participants_info, interp_PMJ_dist_csv, final_csv_filename):
    """
    This function interpolates PMJ distances at 0.1 intervals between each each vertebral level using PMJ distances interpolated between each disc label
    Then, morphometrics are computed for each distance (1.0, 1.1, 1.2, etc.), and the results are saved to a final CSV file.

    Args:
        subject: participant ID 
        output_csv_path: Path to the folder containing the CSV files
        PMJ_distances_csv: CSV file containing PMJ distances for each disc label (generated by the `get_disc_label_PMJ_dist` function above).
        pmj: Path to the PMJ label file
        t2w_seg_file: Path to the T2w SC segmentation file
        participants_info: Path to the participants.tsv file
        interp_PMJ_dist_csv: The output CSV file which will contain the interpolated PMJ distances
        final_csv_filename: The output CSV file containing computed morphometrics for each interpolated PMJ distance

    Outputs:
        subject}_PMJ_dist_interp.csv : CSV files with the interpolated PMJ distances
        {subject}_temp_pmj_{level}.csv: Temporary CSV files for each interpolated PMJ distance
        {subject}_interpolated_morphometrics.csv : A combined CSV containing CSA values for each interpolated distance. This final CSV also
        contains age and sex information, taken from the `participants.tsv` file.

    """

    csv_file = pd.read_csv(PMJ_distances_csv)

    if 'DistancePMJ' in csv_file.columns:
        levels = csv_file['Level'].values
        pmj_distances = csv_file['DistancePMJ'].values

        # Create interpolation function
        interp_func = interp1d(levels, pmj_distances, kind='linear', fill_value='extrapolate')

        # Generate 0.1 step levels between each integer level (excluding integer levels)
        interp_levels = []
        for i in range(int(levels.min()), int(levels.max())):
            interp_levels.extend(np.round(np.arange(i + 0.1, i + 1.0, 0.1), 1))

        interp_levels = np.array(interp_levels)
        interp_distances = interp_func(interp_levels)

        # Create interpolated DataFrame
        df_interp = pd.DataFrame({
            'Level': interp_levels,
            'DistancePMJ': interp_distances
        })

        # Add other morphometrics as NaN for interpolated levels
        morphometric_cols = [col for col in csv_file.columns if col not in ['Level', 'DistancePMJ']]
        for col in morphometric_cols:
            df_interp[col] = np.nan

        # Combine original + interpolated
        df_combined = pd.concat([csv_file, df_interp], ignore_index=True).sort_values(by='Level').reset_index(drop=True)

        # Re-add subject column
        if 'subject' not in df_combined.columns:
            df_combined['subject'] = subject

        # Save final merged morphometrics with interpolated PMJ distances
        df_combined.to_csv(interp_PMJ_dist_csv, index=False)
        print(f"Updated morphometrics CSV with interpolated PMJ distances saved to:\n{interp_PMJ_dist_csv}")
    else:
        print(f"Missing DistancePMJ column — interpolation skipped.")

    # Create a list to store DataFrames for each distance from PMJ 
    all_results = []

    # Read the interpolated CSV file
    csv_file_interpolated = pd.read_csv(interp_PMJ_dist_csv)

    # Merge age and sex into the interpolated CSV before processing
    df_participants_info = pd.read_csv(participants_info, sep='\t').rename(columns={'participant_id': 'subject'})
    csv_file_interpolated = csv_file_interpolated.merge(df_participants_info[['subject', 'age', 'sex']], on='subject', how='left')

    # Iterate through each row in the interpolated CSV file and compute morphometrics for each PMJ distance (each VertLevel interval)
    for index, row in csv_file_interpolated.iterrows():
        level = row['Level']
        distance_pmj = row['DistancePMJ']

        # Process only if the PMJ distance is not none
        if pd.notna(distance_pmj):

            # Create a temporary file to store the result for each vert_level
            temp_csv_filename = os.path.join(output_csv_path, f"{subject}_temp_pmj_{level}.csv")

            if os.path.exists(temp_csv_filename):
                print(f"Temporary file already exists: {temp_csv_filename}. Skipping processing for {level} (PMJ distance: {distance_pmj})")
                continue

            # Call sct_process_segmentation
            sct_process_segmentation.main([
                '-i', t2w_seg_file,
                '-pmj', pmj,
                '-pmj-distance', str(distance_pmj),
                '-pmj-extent', '3',
                '-perlevel', '1',
                '-o', temp_csv_filename,
            ])

            # Read results from the temporary CSV file and append the values to a list (`all_results`)
            temp_df = pd.read_csv(temp_csv_filename)

            # Add the level to the 'VertLevel' column in the temporary CSV file
            temp_df['VertLevel'] == level
            
            # Rename columns
            temp_df = temp_df.rename(columns={
                'MEAN(area)': 'CSA',
                'MEAN(diameter_AP)': 'AP_diameter',
                'MEAN(diameter_RL)': 'RL_diameter',
                'MEAN(eccentricity)': 'eccentricity',
                'MEAN(solidity)': 'solidity'
            })

            # Add subject, level type ('VertLevel' or 'SpinalLevel'), and DistancePMJ columns
            temp_df['VertLevel'] = level
            temp_df['DistancePMJ'] = distance_pmj
            temp_df['subject'] = subject

            # Append to the results list (this list will then contain all morphometrics for each PMJ distance, i.e. from each temp CSV file)
            all_results.append(temp_df[['subject', 'VertLevel', 'DistancePMJ', 'CSA', 'AP_diameter', 'RL_diameter', 'eccentricity', 'solidity']])
            print(f"Processed VertLevel {level} (PMJ distance : {distance_pmj})")

    # Save all results to a final CSV file
    final_results_df = pd.concat(all_results, ignore_index=True)
    
    # Add age and sex columns to the final CSV file
    final_results_df = final_results_df.merge(df_participants_info[['subject', 'age', 'sex']], on='subject', how='left')
    
    # Save the final results to a CSV file
    final_results_df.to_csv(final_csv_filename, index=False)
    print(f"Final morphometrics results saved to:\n{final_csv_filename}")

    # Delete all temporary files
    temp_files_pattern = os.path.join(output_csv_path, f"{subject}_temp_pmj_*.csv")
    temp_files = glob.glob(temp_files_pattern)

    for temp_file in temp_files:
        os.remove(temp_file)
        print(f"Deleted temp file: {temp_file}")


def PMJ_SCtip_normalization(per_slice_csv):

    perslice_df = pd.read_csv(per_slice_csv)

    # Keep slices where CSA > 0 
    SC_slices = perslice_df[perslice_df["MEAN(area)"] > 0]

    # Get the first slice with CSA values (which corresponds to the tip of the SC)
    slice_SC_tip = SC_slices["Slice (I->S)"].min()

    print(f'Slice corresponding to SC tip : {slice_SC_tip}')

    # Get the PMJ distance of the SC tip
    dist_PMJ_SC_tip = perslice_df.loc[perslice_df["Slice (I->S)"] == slice_SC_tip, "DistancePMJ"].values[0]

    # Normalize distances between PMJ and the SC tip
    perslice_df["Normalized_PMJ_SCtip"] = perslice_df["DistancePMJ"] / dist_PMJ_SC_tip

    # Add the "NormalizedDistance" column to the per_slice_csv
    if "Normalized_PMJ_SCtip" not in perslice_df.columns:
        perslice_df.to_csv(per_slice_csv, index=False)
        print(f"Normalized morphometrics results saved to:\n{per_slice_csv}")

def C1_SCtip_normalization(per_slice_csv, PMJ_distances_csv):

    perslice_df = pd.read_csv(per_slice_csv)
    pmj_dist_df = pd.read_csv(PMJ_distances_csv)

    # Keep slices where CSA > 0 
    SC_slices = perslice_df[perslice_df["MEAN(area)"] > 0]

    # Get the first slice with CSA values (which corresponds to the tip of the SC)
    slice_SC_tip = SC_slices["Slice (I->S)"].min()

    # Get the PMJ distance of the SC tip
    dist_PMJ_SC_tip = perslice_df.loc[perslice_df["Slice (I->S)"] == slice_SC_tip, "DistancePMJ"].values[0]

    print(f'Slice corresponding to SC tip : {slice_SC_tip}')

    # Get the PMJ distance corresponding to the C1 vertebrae
    dist_PMJ_C1 = pmj_dist_df.loc[pmj_dist_df["Level"] == 1.0, "DistancePMJ"].values[0]

    print(f'Distance from PMJ to C1: {dist_PMJ_C1}')

    # Normalize distances between C1 vertebrae and the SC tip
    perslice_df["Normalized_C1_SCtip"] = ((perslice_df["DistancePMJ"] - dist_PMJ_C1) / (dist_PMJ_SC_tip - dist_PMJ_C1))

    # Add the "NormalizedDistance" column to the per_slice_csv if it does not already exist
    perslice_df.to_csv(per_slice_csv, index=False)
    print(f"Final normalized morphometrics saved to:\n{per_slice_csv}")


def C1_CE_LE_SCtip_normalization(per_slice_csv, PMJ_distances_csv):

    """
    Normalize across the following SC landmarks : 
    0.0 ---> C1
    0.2 ---> cervical enlargment (maxiumum CSA in the first half of the CSA values)
    0.9 ---> lumbar enlargment (maxiumum CSA in the second half of the CSA values)
    1.0 ---> SC tip 
    """

    perslice_df = pd.read_csv(per_slice_csv)
    pmj_dist_df = pd.read_csv(PMJ_distances_csv)

    # Only keep slices where CSA > 0
    SC_slices = perslice_df[perslice_df["MEAN(area)"] > 0].copy()
    SC_slices = SC_slices.sort_values("Slice (I->S)")

    # ---- 0.0 :  C1 LEVEL ----
    dist_PMJ_C1 = pmj_dist_df.loc[pmj_dist_df["Level"] == 1.0, "DistancePMJ"].values[0]
    print(f"Distance from PMJ to C1: {dist_PMJ_C1}")

    # ---- 1.0 : SC TIP ----
    slice_SC_tip = SC_slices["Slice (I->S)"].min()
    dist_PMJ_SC_tip = perslice_df.loc[perslice_df["Slice (I->S)"] == slice_SC_tip, "DistancePMJ"].values[0]
    print(f"Slice corresponding to SC tip: {slice_SC_tip}")

    # ---- 0.2 CERVICAL ENLARGMENT ----
    
    # Get slice numbers for C2 and C7
    slice_C2 = pmj_dist_df.loc[pmj_dist_df["Level"] == 2.0, "Slice (I->S)"].values[0]
    slice_C7 = pmj_dist_df.loc[pmj_dist_df["Level"] == 7.0, "Slice (I->S)"].values[0]

    # Select slices between C2 and C7
    slice_range_cervical_enlargement = perslice_df[(perslice_df["Slice (I->S)"] >= slice_C7) & (perslice_df["Slice (I->S)"] <= slice_C2)]

    # Find maximum CSA within this range
    idx_max = slice_range_cervical_enlargement["MEAN(area)"].idxmax()
    cervical_row = slice_range_cervical_enlargement.loc[idx_max]
    dist_PMJ_cervical_enlargment = cervical_row["DistancePMJ"]

    print(f"Cervical enlargement distance from PMJ: {dist_PMJ_cervical_enlargment}")

    # ---- 0.9 LUMBAR ENLARGMENT ----

    # Get slice numbers for T5 (level = 12.0) and L1 (level = 20.0)
    slice_T5 = pmj_dist_df.loc[pmj_dist_df["Level"] == 12.0, "Slice (I->S)"].values[0]

    # Select slices between T5 and L1
    slice_range_lumbar_enlargement = perslice_df[(perslice_df["Slice (I->S)"] >= slice_SC_tip) & (perslice_df["Slice (I->S)"] <= slice_T5)]

    # Find maximum CSA within this range
    idx_max = slice_range_lumbar_enlargement["MEAN(area)"].idxmax()
    lumbar_row = slice_range_lumbar_enlargement.loc[idx_max]
    dist_PMJ_lumbar_enlargment = lumbar_row["DistancePMJ"]

    print(f"Lumbar enlargement distance from PMJ: {dist_PMJ_lumbar_enlargment}")

    # ---- 1.0 : SC TIP ----
    slice_SC_tip = SC_slices["Slice (I->S)"].min()
    dist_PMJ_SC_tip = perslice_df.loc[perslice_df["Slice (I->S)"] == slice_SC_tip, "DistancePMJ"].values[0]
    print(f"Slice corresponding to SC tip: {slice_SC_tip}")

    # ---- LANDMARK POSITIONS ----
    landmarks_dist = np.array([
        dist_PMJ_C1,
        dist_PMJ_cervical_enlargment,
        dist_PMJ_lumbar_enlargment,
        dist_PMJ_SC_tip
    ])

    landmarks_norm = np.array([
        0.0,
        0.2,
        0.9,
        1.0
    ])

    # ---- PIECEWISE LINEAR NORMALIZATION ----
    perslice_df["Normalized_C1_CE_LE_SCtip"] = np.interp(
        perslice_df["DistancePMJ"],
        landmarks_dist,
        landmarks_norm
    )

    # Save updated CSV
    perslice_df.to_csv(per_slice_csv, index=False)

    print(f"Final normalizedmorphometrics saved to:\n{per_slice_csv}")

def PMJ_CE_LE_SCtip_normalization(per_slice_csv, PMJ_distances_csv):

    """
    Normalize across the following SC landmarks : 
    0.0 ---> PMJ
    0.2 ---> cervical enlargment (maxiumum CSA in the first half of the CSA values)
    0.9 ---> lumbar enlargment (maxiumum CSA in the second half of the CSA values)
    1.0 ---> SC tip 
    """

    perslice_df = pd.read_csv(per_slice_csv)
    pmj_dist_df = pd.read_csv(PMJ_distances_csv)

    # Only keep slices where CSA > 0
    SC_slices = perslice_df[perslice_df["MEAN(area)"] > 0].copy()
    SC_slices = SC_slices.sort_values("Slice (I->S)")

    # ---- 1.0 : SC TIP ----
    slice_SC_tip = SC_slices["Slice (I->S)"].min()
    dist_PMJ_SC_tip = perslice_df.loc[perslice_df["Slice (I->S)"] == slice_SC_tip, "DistancePMJ"].values[0]
    print(f"Slice corresponding to SC tip: {slice_SC_tip}")

    # ---- 0.2 CERVICAL ENLARGMENT ----
    
    # Get slice numbers for C2 and C7
    slice_C2 = pmj_dist_df.loc[pmj_dist_df["Level"] == 2.0, "Slice (I->S)"].values[0]
    slice_C7 = pmj_dist_df.loc[pmj_dist_df["Level"] == 7.0, "Slice (I->S)"].values[0]

    # Select slices between C2 and C7
    slice_range_cervical_enlargement = perslice_df[(perslice_df["Slice (I->S)"] >= slice_C7) & (perslice_df["Slice (I->S)"] <= slice_C2)]

    # Find maximum CSA within this range
    idx_max = slice_range_cervical_enlargement["MEAN(area)"].idxmax()
    cervical_row = slice_range_cervical_enlargement.loc[idx_max]
    dist_PMJ_cervical_enlargment = cervical_row["DistancePMJ"]

    print(f"Cervical enlargement distance from PMJ: {dist_PMJ_cervical_enlargment}")

    # ---- 0.9 LUMBAR ENLARGMENT ----

    # Get slice numbers for T5 (level = 12.0) and L1 (level = 20.0)
    slice_T5 = pmj_dist_df.loc[pmj_dist_df["Level"] == 12.0, "Slice (I->S)"].values[0]

    # Select slices between T5 and L1
    slice_range_lumbar_enlargement = perslice_df[(perslice_df["Slice (I->S)"] >= slice_SC_tip) & (perslice_df["Slice (I->S)"] <= slice_T5)]

    # Find maximum CSA within this range
    idx_max = slice_range_lumbar_enlargement["MEAN(area)"].idxmax()
    lumbar_row = slice_range_lumbar_enlargement.loc[idx_max]
    dist_PMJ_lumbar_enlargment = lumbar_row["DistancePMJ"]

    print(f"Lumbar enlargement distance from PMJ: {dist_PMJ_lumbar_enlargment}")

    # ---- 1.0 : SC TIP ----
    slice_SC_tip = SC_slices["Slice (I->S)"].min()
    dist_PMJ_SC_tip = perslice_df.loc[perslice_df["Slice (I->S)"] == slice_SC_tip, "DistancePMJ"].values[0]
    print(f"Slice corresponding to SC tip: {slice_SC_tip}")

    # ---- LANDMARK POSITIONS ----
    landmarks_dist = np.array([
        0,
        dist_PMJ_cervical_enlargment,
        dist_PMJ_lumbar_enlargment,
        dist_PMJ_SC_tip
    ])

    landmarks_norm = np.array([
        0.0,
        0.2,
        0.9,
        1.0
    ])

    # ---- PIECEWISE LINEAR NORMALIZATION ----
    perslice_df["Normalized_PMJ_CE_LE_SCtip"] = np.interp(
        perslice_df["DistancePMJ"],
        landmarks_dist,
        landmarks_norm
    )

    # Save updated CSV
    perslice_df.to_csv(per_slice_csv, index=False)

    print(f"Final normalizedmorphometrics saved to:\n{per_slice_csv}")


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

    # Check if final interpolated morphometrics CSV file already exists 
    if os.path.exists(final_csv):
        print(f"Interpolated morphometrics already exists for subject {subject}: {final_csv}. Skipping.")
    else:
        print(f"Computing interpolated morphometrics subject: {subject}")

        # Step 1 : Run sct_process_segmentation per slice to get the PMJ distances of each slice
        run_sct_process_segmentation_per_slice(
            pmj=t2w_pmj_label,
            t2w_seg_file=t2w_seg_file,
            output_per_slice_csv=output_per_slice_csv
            )


        # Step 2 : Get the disc label slices and add the PMJ distances of each disc label
        get_disc_label_PMJ_dist(
            subject, 
            output_per_slice_csv,
            output_csv_dir, 
            t2w_disc_labels,
            output_PMJ_dist_csv=output_PMJ_dist_csv)
        
        # Step 3 : Interpolate the PMJ distances and run sct_process_segmentation for all interpolated PMJ distances
        compute_interpolated_morphometrics(
            subject,
            output_csv_dir, 
            PMJ_distances_csv=output_PMJ_dist_csv, 
            pmj=t2w_pmj_label, 
            t2w_seg_file=t2w_seg_file, 
            participants_info=participants_info, 
            interp_PMJ_dist_csv=interp_PMJ_dist_csv,
            final_csv_filename=final_csv
            )
        
    print(f"Normalizing SC with PMJ and SC tip for: {subject}")
    
    # Step 4 : Normalize the distances using spinal cord landmarks (PMJ, C1, cervical and lumbar enlargments, tip of SC)

    # Normalize with PMJ and SC tip only
    PMJ_SCtip_normalization(output_per_slice_csv)

    # Normalize the distances between C1 and SC tip
    C1_SCtip_normalization(output_per_slice_csv, output_PMJ_dist_csv)

    # Normalize the distances between C1, cervical enlargment, lumbar enlargment and SC tip
    C1_CE_LE_SCtip_normalization(output_per_slice_csv, output_PMJ_dist_csv)

    # Normalize the distances between PMJ, cervical enlargment, lumbar enlargment and SC tip
    PMJ_CE_LE_SCtip_normalization(output_per_slice_csv, output_PMJ_dist_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run morphometric extraction for one subject")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001)")
    parser.add_argument("--data-path", required=True, help="Path to raw data")
    parser.add_argument("--path-output", required=True, help="Path to output results")
    parser.add_argument("--subject-dir", required=True, help="Path to subject folder (e.g., sub-001)")
    parser.add_argument("--file-t2", required=True, help="T2-weighted image prefix (e.g., sub-01_T2w)")

    args = parser.parse_args()

    main(args.subject, args.data_path, args.path_output, args.subject_dir, args.file_t2)
