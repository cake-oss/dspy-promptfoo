#!/usr/bin/bash

#install DSPy
pip install -q dspy

#install promptfoo
npm install -q promptfoo@latest

PATH=./node_modules/.bin:/opt/conda/bin:/opt/conda/condabin:/command:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

promptfoo eval
