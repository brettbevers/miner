#!/usr/bin/env bash

pip install setuptools==33.1.1

if [ "$(ls -A $1/*.egg)" ]; then
     for e in "$1/*.egg"; do easy_install -Z $e; done;
else
    echo "$1 is Empty"
fi