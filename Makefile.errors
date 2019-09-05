all: csvs errors images plots

images: uncompressed.png decompressed-t-0.png decompressed-t-8.png decompressed-t-15.png decompressed-p-6.png decompressed-p-10.png decompressed-p-19.png

csvs: precision.csv tolerance.csv rate.csv

plots: plots1 plotsn plotsm

plots1: plot-tolerance-cf.png plot-tolerance-err-norm.png plot-tolerance-err-max.png plot-precision-cf.png plot-precision-err-norm.png plot-precision-err-max.png

plotsn: plotn-error-decompressed-t-0.csv plotn-error-decompressed-t-8.csv plotn-error-decompressed-t-15.csv plotn-error-decompressed-p-6.csv plotn-error-decompressed-p-10.csv plotn-error-decompressed-p-19.csv

plotsm: plotm-error-decompressed-t-0.csv plotm-error-decompressed-t-8.csv plotm-error-decompressed-t-15.csv plotm-error-decompressed-p-6.csv plotm-error-decompressed-p-10.csv plotm-error-decompressed-p-19.csv

errors: error-decompressed-t-0.csv error-decompressed-t-8.csv error-decompressed-t-15.csv error-decompressed-p-6.csv error-decompressed-p-10.csv error-decompressed-p-19.csv

plotn-%.png: %.csv plotter.py
	python plotter.py $< 0 1 "Progression of error norm with simulation time" $@

plotm-%.png: %.csv plotter.py
	python plotter.py $< 0 2 "Progression of error norm with simulation time" $@

plot-tolerance-cf.png: tolerance.csv plotter.py 
	python plotter.py tolerance.csv 4 1 "Compression factor for varying tolerance" $@

plot-tolerance-err-norm.png: tolerance.csv plotter.py 
	python plotter.py tolerance.csv 4 5 "Error norm for varying tolerance" $@

plot-tolerance-err-max.png: tolerance.csv plotter.py 
	python plotter.py tolerance.csv 4 6 "Maximum error for varying tolerance" $@

plot-precision-cf.png: precision.csv plotter.py 
	python plotter.py precision.csv 4 1 "Compression factor for varying precision" $@

plot-precision-err-norm.png: precision.csv plotter.py 
	python plotter.py precision.csv 4 5 "Error norm for varying precision" $@

plot-precision-err-max.png: precision.csv plotter.py 
	python plotter.py precision.csv 4 6 "Maximum error for varying precision" $@

plot-prog-norm-p.png: 

error-%.csv: %.h5 uncompressed.h5 difference.py
	python -u difference.py uncompressed.h5 $< | tee $@

%.png: %.h5
	python plot_numpy_hdf5.py $< $@

precision.csv: uncompressed.h5 precision.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python precision.py uncompressed.h5 > precision.csv

tolerance.csv: uncompressed.h5 tolerance.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python tolerance.py uncompressed.h5 > tolerance.csv

rate.csv: uncompressed.h5 rate.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python rate.py uncompressed.h5 > rate.csv

uncompressed.h5: overthrust_3D_initial_model.h5 zfp-0.5.3/lib/libzfp.so simple.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib DEVITO_OPENMP=1 python simple.py

zfp-0.5.3/lib/libzfp.so: zfp-0.5.3.tar.gz
	tar -xzvf zfp-0.5.3.tar.gz
	cd zfp-0.5.3 && make && make shared

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

zfp-0.5.3.tar.gz: 
	wget https://computation.llnl.gov/projects/floating-point-compression/download/zfp-0.5.3.tar.gz

