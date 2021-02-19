#!/bin/env bash

base_dir="$HOME/astro/repositories/github/CorrMatrix_ABC/scripts/ABC/WL"

files=(cosmo.par cosmo_lens.par nofz_Euclid_1bin.par nofz_Euclid_1bin.txt)
n=${#files[@]}

arg="$1"
case "$arg" in
	"ABC")
		files[$n]=wl_functions.py
		cp $base_dir/wl_model.input .
		;;
	"PMC")
		;;
	*)
		echo "Invalid mode, one of [ABC|PMC]"
		exit 1
esac


# Set links
for f in ${files[*]}; do
	source=$base_dir/$f
	ln -sf $source
done

mv nofz_Euclid_1bin.par nofz.par



