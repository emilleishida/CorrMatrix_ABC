#!/bin/env bash

### Post-process ABC sample

num=$1

echo "Post-processing ABC iteration $num"

echo "Creating pmc-compatible simulation file"

# Add weights to ABC sample file
echo "# weight" > tmp.tmp
cat linear_PS${num}weights.dat >> tmp.tmp
paste tmp.tmp linear_PS${num}.dat > linear_PS${num}+weight.dat
rm tmp.tmp

# Create pmc simulation file
ascii2sample.pl -w 0 -t LIN -c '1 2' -s 'Omega_m sigma_8' -N 1 linear_PS${num}+weight.dat

# Mean and rms
echo "Mean and rms"
meanvar_sample linear_PS${num}+weight.dat.sample -c ~/astro/Runs/ABC/PMC/WL/Euclid_2par_1bin/config_pmc | tee mean_${num}

# Plotting
echo "Resample to create sample file"
Rscript ~/astro/ECOSSTAT-work/R/sample_from_pmcsimu.R -M 2 -i linear_PS${num}+weight.dat.sample -o sample_${num}

echo "1D and 2D plots"
Rscript $COSMOPMC/R/plot_confidence.R sample_${num} -c ~/astro/Runs/ABC/PMC/WL/Euclid_2par_1bin/config_pmc -o pdf --pmin=0.1_0.6 --pmax=0.5_1.0 -N 200 --gsmooth=70 -C 1

