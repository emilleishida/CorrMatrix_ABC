#!/bin/env bash

base_dir="$HOME/astro/repositories/github/CorrMatrix_ABC/scripts/ABC/WL"

files=(cosmo.par cosmo_lens.par nofz_Euclid_1bin.par nofz_Euclid_1bin.txt)
n=${#files[@]}

arg="$1"
case "$arg" in
	"ABC")
		files[$n]=wl_functions.py
        n=$((n+1))
        file[$n]=abc_wl.py
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

ln -sf $base_dir/Cov_SSC/cov_SSC_rel_lin_10.txt cov_SSC_rel_lin.txt

if [[ "$arg" == "ABC" ]]; then
    mv wl_functions.py toy_model_functions.py
    mv abc_wl.py ABC_est_cov.py
    echo "Now create, update, or copy toy model input file, e.g."
	echo "cp $base_dir/wl_model.input ./toy_model.input"
fi
