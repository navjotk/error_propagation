all: stream.txt csvs runtime-ot2-32.txt runtime-ot4-32.txt runtime-tti-32.txt

stream.txt: stream.o
	./stream.o > stream.txt

stream.o: stream.c
	gcc -fopenmp -O3 -o stream.o stream.c

runtime.txt: uncompressed.h5

uncompressed.h5: overthrust_3D_initial_model.h5 simple.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib DEVITO_OPENMP=1 python -u simple.py -so 32 > runtime-ot2-32.txt 2>&1

runtime-ot4-32.txt: overthrust_3D_initial_model.h5 zfp-0.5.3/lib/libzfp.so simple.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib DEVITO_OPENMP=1 python -u simple.py -so 32 -k OT4 > runtime-ot4-32.txt 2>&1

runtime-tti-32.txt: overthrust_3D_initial_model.h5 zfp-0.5.3/lib/libzfp.so simple.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib DEVITO_OPENMP=1 python -u simple.py -so 32 -k TTI > runtime-tti-32.txt 2>&1

csvs: tolerance-zfp.csv tolerance-sz.csv lossless.csv diffcsvs

diffcsvs: lossless-diff-sz-0.csv lossless-diff-sz-8.csv lossless-diff-sz-16.csv lossless-diff-zfp-0.csv lossless-diff-zfp-8.csv lossless-diff-zfp-16.csv

precision.csv: uncompressed.h5 precision.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u precision.py uncompressed.h5 > precision.csv

tolerance-zfp.csv: uncompressed.h5 tolerance.py
	python -u tolerance.py uncompressed.h5 zfp > tolerance-zfp.csv

tolerance-sz.csv: uncompressed.h5 tolerance.py
	python -u tolerance.py uncompressed.h5 sz > tolerance-sz.csv

lossless.csv: uncompressed.h5 lossless.py
	python -u lossless.py uncompressed.h5 > lossless.csv

lossless-diff-sz-0.csv: tolerance-sz.csv lossless.py
	python -u lossless.py error_field-sz-0.h5 > lossless-diff-sz-0.csv

lossless-diff-sz-8.csv: tolerance-sz.csv lossless.py
	python -u lossless.py error_field-sz-8.h5 > lossless-diff-sz-8.csv

lossless-diff-sz-16.csv: tolerance-sz.csv lossless.py
	python -u lossless.py error_field-sz-16.h5 > lossless-diff-sz-16.csv

lossless-diff-zfp-0.csv: tolerance-zfp.csv lossless.py
	python -u lossless.py error_field-zfp-0.h5 > lossless-diff-zfp-0.csv

lossless-diff-zfp-8.csv: tolerance-zfp.csv lossless.py
	python -u lossless.py error_field-zfp-8.h5 > lossless-diff-zfp-8.csv

lossless-diff-zfp-16.csv: tolerance-zfp.csv lossless.py
	python -u lossless.py error_field-zfp-16.h5 > lossless-diff-sz-16.csv

rate.csv: uncompressed.h5 rate.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u rate.py uncompressed.h5 > rate.csv

precision-s.csv: uncompressed.h5 precision.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u precision.py uncompressed.h5 --no-parallel > precision-s.csv

tolerance-s.csv: uncompressed.h5 tolerance.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u tolerance.py uncompressed.h5 --no-parallel > tolerance-s.csv

rate-s.csv: uncompressed.h5 rate.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u rate.py uncompressed.h5 --no-parallel > rate-s.csv


overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5
