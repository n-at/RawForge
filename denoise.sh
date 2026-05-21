#!/bin/bash

source venv/bin/activate

for file in input/*.dng; do
    echo "processing ${file}"
    file_output="${file}.denoised.dng"
    python -m RawForge.main --in_file "${file}" --out_file "${file_output}" $@
    if [ ! -z $convert ]; then
        convert "${file_output}" "${file}.denoised.${convert}"
        rm "${file_output}"
    fi
done
