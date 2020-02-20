#!/bin/env bash

### Post-process ABC sample, non-WL version (newer)

num=$1

pref="quad"
par1="t"
par2="A"
PMC="$HOME/astro/repositories/github/CosmoPMC"

echo "Post-processing ABC iteration $num"

echo "Creating pmc-compatible simulation file"

# Add weights to ABC sample file
echo "# weight" > tmp.tmp
cat ${pref}_PS${num}weights.dat >> tmp.tmp
paste tmp.tmp ${pref}_PS${num}.dat > ${pref}_PS${num}+weight.dat
rm tmp.tmp

# Create pmc simulation file
$PMC/bin/ascii2sample.pl -w 0 -t LIN -c '1 2' -s '$par1 $par2' -N 1 ${pref}_PS${num}+weight.dat

# Mean and rms
#echo "Mean and rms"
#$PMC/bin/meanvar_sample ${pref}_PS${num}+weight.dat.sample -c ~/astro/repositories/github/CorrMatrix_ABC/scripts/ABC/templates_quad/config_pmc | tee mean_${num}

# Plotting
echo "Resample to create sample file"
Rscript $PMC/R/sample_from_pmcsimu.R -M 2 -i ${pref}_PS${num}+weight.dat.sample -o sample_${num}

echo "1D and 2D plots"
Rscript $PMC/R/plot_confidence.R sample_${num} -c ~/astro/Runs/ABC/PMC/WL/Euclid_2par_1bin/config_pmc -o pdf --pmin=0.1_0.6 --pmax=0.5_1.0 -N 200 --gsmooth=70 -C 1

