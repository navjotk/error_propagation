from argparse import ArgumentParser
import h5py
import numpy as np
import matplotlib.pyplot as plt


description = ("Slice a 3D Model into a 2D model")
parser = ArgumentParser(description=description)
parser.add_argument("--filename", type=str, required=True)
parser.add_argument("--datakey", type=str, default="m")
parser.add_argument("--slice-loc", type=int, default=401)
parser.add_argument("--dimension-order", choices=["xyz", "xzy"], default="xzy")
args = parser.parse_args()

filename = args.filename
slice_loc = args.slice_loc
datakey = args.datakey
dimension_order = args.dimension_order

basename, extension = filename.split(".")

outfilename = basename+"_2D."+extension

with h5py.File(filename, 'r') as ifile:
    data_m = ifile[datakey][()]
    if dimension_order == "xzy":
        m_sliced = data_m[:, :, slice_loc]
    else:
        m_sliced = data_m[:, slice_loc, :]
    try:
        data_o = ifile['o'][()]
        data_d = ifile['d'][()]
        o_sliced = data_o[:2]
        d_sliced = data_d[:2]
        write_metadata = True
    except KeyError:
        write_metadata = False
    
    other_elements_to_copy = [x for x in list(ifile.keys()) if x != datakey]

    with h5py.File(outfilename, 'w') as ofile:
        ofile.create_dataset(datakey, data=m_sliced)
        if write_metadata:
            ofile.create_dataset('o', data=o_sliced)
            ofile.create_dataset('d', data=d_sliced)

data = m_sliced

shape = data.shape
vmax = np.max(data)

im = plt.imshow(data, vmax=vmax, vmin=0, cmap="GnBu",
                extent=[0, 20, 0.001*(shape[-1]-1)*25, 0])

plt.xlabel("X (km)")
plt.ylabel("Depth (km)")
cb = plt.colorbar(shrink=.3, pad=.01, aspect=10)
for i in cb.ax.yaxis.get_ticklabels():
    i.set_fontsize(12)

    cb.set_label('Pressure')

# plt.savefig(output_file, bbox_inches='tight')
plt.show()
plt.clf()
