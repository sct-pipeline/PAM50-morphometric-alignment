import sys, os
import argparse
import glob
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import spinalcordtoolbox.utils as sct
from spinalcordtoolbox.scripts import sct_process_segmentation

"""
This script computes spinal cord morphometrics (e.g., CSA) from T2-weighted data.

The script first computes morphometrics per slice and per level. 
Then, it interpolates PMJ distances for each Vertebral Level, and computes morphometrics at each interpolated PMJ distance (at 0.1 mm intervals between each Vertebral Level).

Outputs :
- Temporary CSV files are created for each interpolated PMJ distance (stored inside `output/results` from your `path-output` folder defined in the config file)
- A combined CSV containing CSA values for each interpolated distance will be saved per subject under `results/morphometrics`
- The final CSV files will contain age and sex information for each subject, taken from the `participants.tsv` file.

Usage : 
    The script can be run with `sct_run_batch` using the wrapper script `wrapper_rootlets.sh` as follows:
        
        sct_run_batch -config config/config.yaml -script wrapper/wrapper_extract_morphometrics.sh

Author: Samuelle St-Onge
"""

def compute_morphometrics(output_csv_filename, participants_info, t2w_seg_file, label_file, perlevel=1, perslice=1):
    """
    This function computes spinal cord morphometrics using `sct_process_segmentation`. 
    The output morphometrics are saved in a CSV file with the age and sex info for each subject, taken from the 1participants.tsv` file. 
    
    Args : 
        output_csv_filename: Name of the output CSV file
        participants_info: Path to the `participants.tsv` file
        t2w_seg_file: Path to the spinal cord segmentation mask
        disc_file: The labeled discs
        perlevel: Output either one metric per level (perlevel=1) or a single output metric for all levels (perlevel=0)
        perslice: Output either one metric per slice (perslice=1) or a single output metric for all slices (perslice=0)
        pmj: Path to the PMJ label
    """
    # Run sct_process_segmentation
    sct_process_segmentation.main([
        '-i', t2w_seg_file,
        '-discfile', label_file,
        '-perlevel', perlevel,
        '-perslice', perslice,
        '-o', output_csv_filename
    ])

    # Load results in output CSV file 
    df = pd.read_csv(output_csv_filename)

    # Get the subject ID from the filename
    df['subject'] = 'sub-' + df['Filename'].astype(str).str.extract(r'sub-([0-9]+)')[0]

    # Get the age and sex from the `participants.tsv`` file, and add to the morphometrics CSV file 
    df_age = pd.read_csv(participants_info, sep='\t').rename(columns={'participant_id': 'subject'})
    df.columns = df.columns.str.strip()
    df_age.columns = df_age.columns.str.strip()

    df = df.drop(columns=[col for col in ['age', 'sex'] if col in df.columns])
    df_merged = df.merge(df_age[['subject', 'age', 'sex']], on='subject', how='left')

    # Rename columns and save the changes to the CSV file
    df_merged = df_merged.rename(columns={
        'MEAN(area)': 'CSA',
        'MEAN(diameter_AP)': 'AP_diameter',
        'MEAN(diameter_RL)': 'RL_diameter',
        'MEAN(eccentricity)': 'eccentricity',
        'MEAN(solidity)': 'solidity'
    })

    df_merged.to_csv(output_csv_filename, index=False)


def main(subject, data_path, path_output, subject_dir, file_t2):

    # Define paths
    t2w_seg_file = os.path.join(subject_dir, f"{file_t2}_label-SC_seg.nii.gz")
    t2w_disc_labels = os.path.join(subject_dir, f"{file_t2}_labels-disc-manual.nii.gz")
    participants_info = os.path.join(data_path, 'participants.tsv')

    # Define output CSV file
    output_csv_dir = os.path.join("results/morphometrics")
    os.makedirs(output_csv_dir, exist_ok=True) # Create a folder named "morphometrics" inside the output results folder
    output_csv_filename = os.path.join(output_csv_dir, f"{subject}_morphometrics.csv")


    if os.path.exists(output_csv_filename):
        print(f"Final CSV already exists for subject {subject}: {output_csv_filename}. Skipping processing.")
        return
    else:
        print(f"Processing subject: {subject}")

    # Compute morphometrics per slice and per level
    compute_morphometrics(
        output_csv_filename=os.path.join(output_csv_dir, f"{subject}_morphometrics.csv"),
        participants_info=participants_info,
        t2w_seg_file=t2w_seg_file,
        label_file=t2w_disc_labels, # Use the labeled segmentation 
        perlevel='1',
        perslice='1',
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run morphometric extraction for one subject")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001)")
    parser.add_argument("--data-path", required=True, help="Path to raw data")
    parser.add_argument("--path-output", required=True, help="Path to output results")
    parser.add_argument("--subject-dir", required=True, help="Path to subject folder (e.g., sub-001)")
    parser.add_argument("--file-t2", required=True, help="T2-weighted image prefix (e.g., sub-01_T2w)")

    args = parser.parse_args()

    main(args.subject, args.data_path, args.path_output, args.subject_dir, args.file_t2)
