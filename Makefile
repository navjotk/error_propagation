colon := :
$(colon) := :


all: stream.txt direct

stream.txt: stream.o
	./stream.o > stream.txt

stream.o: stream.c
	gcc -fopenmp -O3 -o stream.o stream.c

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

overthrust_3D_true_model_2D.h5: slicer.py overthrust_3D_true_model.h5
	python slicer.py --filename overthrust_3D_true_model.h5 --datakey m

overthrust_3D_true_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5\

overthrust_3D_initial_model_2D.h5: slicer.py overthrust_3D_initial_model.h5
	python slicer.py --filename overthrust_3D_initial_model.h5 --datakey m0

uncompressed.h5: overthrust_3D_initial_model.h5 simple.py
	DEVITO_OPENMP=1 python simple.py

direct: direct_L0.tex direct_L1.tex direct_L2.tex direct_Linf.tex direct_psnr.tex

direct_%.tex: uncompressed.h5 plots_direct_compression.py
	python plots_direct_compression.py --plot $*

forward: forward_L0.tex

forward_prop_results.csv: forward_error.py overthrust_3D_initial_model.h5
	python forward_error.py

forward_L0.tex: forward_prop_results.csv plot_forward_error.py
	python plot_forward_error.py --filename forward_prop_results.csv

gradient: gradient_error_results.csv plot_gradient_error.py direct_compression_results.csv
 	python plot_gradient_error.py --filename "gradient_error_results.csv" --replacex "direct_compression_results.csv$(:)psnr" --nolog
 	python plot_gradient_error.py --filename "gradient_error_results.csv"
 	python plot_gradient_error.py --filename "gradient_error_results.csv" --replacex "direct_compression_results.csv$(:)cf" --nolog
 	python plot_gradient_multi_nt.py --gradient-results-file gradient_error_results_all.csv

stacking: stacking_experiment.tex

 stacking_experiment_results.csv: stacking.py
 	python stacking.py

 stacking_experiment.tex: stacking_experiment_results.csv
 	python plot_stacking.py --filename stacking_experiment_results.csv

 subsampling: subsampling_experiment.tex

 subsampling_experiment.tex: subsampling_results.csv
 	python plot_subsampling.py --filename subsampling_results.csv

 subsampling_results.csv: subsampling.py
 	rm subsampling_results.csv
 	python subsampling.py

 compressibility: progression_cf_timestep.tex

cf_vs_nt.csv:
 	python compression_ratios_over_time.py

progression_cf_timestep.tex: cf_vs_nt.csv plotter.py
 	python plotter.py --filename cf_vs_nt.csv --basename progression --xvar timestep --yvar cf --no-xlog --no-ylog --hline 1