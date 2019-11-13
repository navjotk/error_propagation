all: stream.txt fwi direct

stream.txt: stream.o
	./stream.o > stream.txt

stream.o: stream.c
	gcc -fopenmp -O3 -o stream.o stream.c

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

fwi: fwi.py shots/shot1.h5 overthrust_3D_initial_model_2D.h5
	python -u fwi.py

shots/shot1.h5: overthrust_3D_true_model_2D.h5 generate_shot_data.py
	python -u generate_shot_data.py


overthrust_3D_true_model_2D.h5: slicer.py overthrust_3D_true_model.h5
	python slicer.py overthrust_3D_true_model.h5

overthrust_3D_true_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5

overthrust_3D_initial_model_2D.h5: slicer.py overthrust_3D_initial_model.h5
	python slicer.py overthrust_3D_initial_model.h5

uncompressed.h5: overthrust_3D_initial_model.h5 simple.py
	DEVITO_OPENMP=1 python simple.py


direct: direct_L0.pdf direct_L1.pdf direct_L2.pdf direct_Linf.pdf direct_psnr.pdf

direct_%.pdf: uncompressed.h5 plots_direct_compression.py
	python plots_direct_compression.py --plot $*

forward: forward_L0.pdf

forward_prop_results.csv: forward_error.py overthrust_3D_initial_model.h5
	python forward_error.py

forward_L0.pdf: forward_prop_results.csv plot_forward_error.py
	python plot_forward_error.py --filename forward_prop_results.csv
