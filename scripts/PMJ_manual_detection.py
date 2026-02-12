#
# Script to identify PMJ labels manually across a dataset
#
# This script was used to identify the PMJ labels across the `whole-spine` dataset
#
# Usage:
#   1. Activate venv_sct
#   2. Run the following script:
#   python scripts/PMJ_manual_detection.py -i /path/to/dataset
#
# Author: Samuelle St-Onge
#

import os
import argparse
import yaml 
import spinalcordtoolbox.utils as sct
from spinalcordtoolbox.scripts import sct_label_utils


def get_arguments():
    parser = argparse.ArgumentParser(description="Manual PMJ detection")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to T2w data")
    return parser.parse_args()

# Get the arguments
args = get_arguments()

# Loop through each subject in the input directory
for subject in os.listdir(args.input_dir):
    print(f'---- PMJ detection for {subject} ----')

    # Check if T2w image exists
    if os.path.exists(os.path.join(args.input_dir, subject, f"anat/{subject}_T2w.nii.gz")):
        T2w_image = os.path.join(args.input_dir, subject, f"anat/{subject}_T2w.nii.gz")    
        PMJ_label = os.path.join(args.input_dir, "derivatives/labels", subject, f"anat/{subject}_T2w_label-PMJ_dlabel.nii.gz")

        # Check if PMJ label is already present. If so, skip. 
        if os.path.exists(PMJ_label):
            continue
        else: 
            sct_label_utils.main(['-i', T2w_image,
                                '-create-viewer', '1', # only 1 label for the PMJ
                                '-o', PMJ_label])
