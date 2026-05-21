#!/bin/bash

if [ ! -x venv ]; then
 python3 -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt

python -m RawForge.download

if [ ! -x input ]; then
    mkdir input
fi
