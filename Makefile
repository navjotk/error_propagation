all: stream.txt

paper.pdf: paper/bookchapter.tex
	cd paper && pdflatex bookchapter.tex 

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
	python slicer.py --filename overthrust_3D_true_model.h5 --datakey m

overthrust_3D_true_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5\

overthrust_3D_initial_model_2D.h5: slicer.py overthrust_3D_initial_model.h5
	python slicer.py --filename overthrust_3D_initial_model.h5 --datakey m0
