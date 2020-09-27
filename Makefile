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

gradient: gradient_error_results.csv
        plot_gradient_error.py --filename gradient_error_results.csv --replacex direct_compression_results.csv:psnr --nolog
