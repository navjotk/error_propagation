from fwi import fwi_gradient, fwi_gradient_checkpointed


if __name__ == "__main__":
    description = ("Experiment to see the effect of stacking on accumulation of errors")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--ncp", default=None, type=int)
    parser.add_argument("--compression", choices=[None, 'zfp', 'sz', 'blosc'], default=None)
    parser.add_argument("--tolerance", default=6, type=int)
    parser.add_argument("--runmode", choices=["error", "timing"], default="timing")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")

    args = parser.parse_args()
    compression_params={'scheme': args.compression, 'tolerance': 10**(-args.tolerance)}

    path_prefix = os.path.dirname(os.path.realpath(__file__))
    compare_error(nbpml=args.nbpml, ncp=args.ncp,
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='%s/overthrust_3D_initial_model.h5'%path_prefix,
        compression_params=compression_params)
