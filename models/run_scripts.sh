#!/bin/bash

py_files=`ls e005*.py`

for f in $py_files
do
	python3 $f
done

