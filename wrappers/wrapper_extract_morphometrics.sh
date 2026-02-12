#!/bin/bash
set -e  # Exit if any command fails

# Wrapper to run `extract_morphometrics.py`` per participant for use with `sct_run_batch`

# Usage : 
#
# cd /path/to/PAM50_morphometric_alignment
#
# sct_run_batch \
#   -config config/config.json
#   -script wrappers/extract_morphometrics.sh \

# Variables provided by sct_run_batch:
# - PATH_DATA
# - PATH_OUTPUT
# - SUBJECT

# Get subject
SUBJECT=$1

echo "Running subject: ${SUBJECT}"
echo "Using data path: ${PATH_DATA}"
echo "Using output path: ${PATH_RESULTS}"

# Check for SUBJECT variable
if [[ -z "${SUBJECT}" ]]; then
    echo "ERROR: SUBJECT variable is not set."
    exit 1
fi

# Define paths
SUBJECT_DIR="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat"

# Define T2 file prefixes for composed and top acquisition T2 files 
T2_FILE_SUFFIX=${SUBJECT}_T2w

# Check if T2w file exists
if [ -f "${PATH_DATA}/${SUBJECT}/anat/${T2_FILE_SUFFIX}.nii.gz" ]; then
    T2_FILE=${T2_FILE_SUFFIX}
else
    echo "T2w file not found for subject ${SUBJECT}. Skipping."
    continue  # Skip to the next subject
fi

# Run extract_morphometrics.py
python "scripts/extract_morphometrics.py" \
    --subject "${SUBJECT}" \
    --data-path "${PATH_DATA}" \
    --path-output "${PATH_RESULTS}" \
    --subject-dir "${SUBJECT_DIR}" \
    --file-t2 "${T2_FILE}" \