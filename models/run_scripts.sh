#!/bin/bash

py_files=`ls e*.py`

for f in $py_files
do
	python3 $f
done

