import argparse
import importlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import zerorpc

DMD_SERVER_IP = '192.168.1.154'
DMD_SERVER_PORT = '6001'

# Paths inside the shot h5 file saved by feedback_atoms.py
RESULTS_GROUP = 'results/feedback_atoms'
PROFILE_X_DATASET = RESULTS_GROUP + '/profile_x_um'
PROFILE_Y_DATASET = RESULTS_GROUP + '/profile_y_sent'
ATOMS_PROFILE_DATASET = RESULTS_GROUP + '/atoms_density_profile_x'
LAST_PROFILE_X_DATASET = RESULTS_GROUP + '/last_profile_x_um'
LAST_PROFILE_Y_DATASET = RESULTS_GROUP + '/last_profile_y'
FLAG_DATASET = RESULTS_GROUP + '/dmd_profile_updated'

# -----------------------------------------------------------------------------
# In-script defaults (edit these when running the script directly from editor)
# CLI arguments still override these values if provided.
# -----------------------------------------------------------------------------
SCRIPT_ARGS = {
    'year': 2026,
    'month': 3,
    'day': 6,
    'sequence': 6,
    'it': 0,
    'rep': 19,
}


def load_modules():
    """Load project helpers without requiring installation as a package."""
    spec_main = importlib.util.spec_from_file_location(
        "general_lib.main",
        "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py",
    )
    general_lib_mod = importlib.util.module_from_spec(spec_main)
    spec_main.loader.exec_module(general_lib_mod)

    spec_settings = importlib.util.spec_from_file_location(
        "general_lib.settings",
        "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py",
    )
    settings_mod = importlib.util.module_from_spec(spec_settings)
    spec_settings.loader.exec_module(settings_mod)

    return general_lib_mod, settings_mod


def parse_args(default_bec2_path):
    parser = argparse.ArgumentParser(
        description='Load a previously saved DMD profile from an old shot h5 file and send it to the DMD server.'
    )
    parser.add_argument('--year', type=int, default=SCRIPT_ARGS['year'])
    parser.add_argument('--month', type=int, default=SCRIPT_ARGS['month'])
    parser.add_argument('--day', type=int, default=SCRIPT_ARGS['day'])
    parser.add_argument('--sequence', type=int, default=SCRIPT_ARGS['sequence'])
    parser.add_argument('--it', type=int, default=SCRIPT_ARGS['it'], help='Iteration number (4 digits in filename).')
    parser.add_argument('--rep', type=int, default=SCRIPT_ARGS['rep'], help='Repetition number. Use 0 for non-rep shots.')
    parser.add_argument('--bec2-path', type=str, default=default_bec2_path)
    args = parser.parse_args()

    required_fields = ['year', 'month', 'day', 'sequence', 'it']
    missing = [field for field in required_fields if getattr(args, field) is None]
    if missing:
        parser.error(
            'Missing required shot identifiers: {}. '\
            'Set them in SCRIPT_ARGS at top of file or pass them via CLI.'.format(', '.join(missing))
        )

    return args


def main():
    general_lib_mod, settings_mod = load_modules()
    args = parse_args(settings_mod.bec2path)
    print('Parsed arguments:', args)

    # 1) Resolve the h5 file from (year, month, day, sequence, it, rep)
    h5_path = general_lib_mod.get_h5filename(
        year=args.year,
        month=args.month,
        day=args.day,
        sequence=args.sequence,
        it=args.it,
        bec2_path=args.bec2_path,
        rep=args.rep,
        show_errs=True,
    )

    if h5_path is None:
        print('Could not find requested h5 file.')
        return

    print('Loading from:', h5_path)

    # 2) Read saved profiles from h5
    with h5py.File(h5_path, 'r') as h5file:
        if PROFILE_X_DATASET not in h5file or PROFILE_Y_DATASET not in h5file:
            print('Saved feedback profile not found in this shot.')
            print('Expected datasets:')
            print('  -', PROFILE_X_DATASET)
            print('  -', PROFILE_Y_DATASET)
            return

        profile_x = np.array(h5file[PROFILE_X_DATASET][...])
        profile_y = np.array(h5file[PROFILE_Y_DATASET][...])

        atoms_profile = None
        if ATOMS_PROFILE_DATASET in h5file:
            atoms_profile = np.array(h5file[ATOMS_PROFILE_DATASET][...])

        last_profile_x = None
        last_profile_y = None
        if LAST_PROFILE_X_DATASET in h5file and LAST_PROFILE_Y_DATASET in h5file:
            last_profile_x = np.array(h5file[LAST_PROFILE_X_DATASET][...])
            last_profile_y = np.array(h5file[LAST_PROFILE_Y_DATASET][...])

        if FLAG_DATASET in h5file:
            already_used = bool(h5file[FLAG_DATASET][()])
            print('Shot marked as used for feedback:', already_used)

    # 3) Show what will be sent
    fig, ax1 = plt.subplots()
    if atoms_profile is not None and atoms_profile.shape == profile_x.shape:
        ax1.plot(profile_x, atoms_profile, 'b-', label='Atoms profile from selected shot')
        ax1.set_ylabel('Atom Density (a.u.)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        if last_profile_y is not None and last_profile_x is not None:
            ax2.plot(last_profile_x, last_profile_y, color='orange', linestyle='--', label='Last profile (before this shot)')
        ax2.plot(profile_x, profile_y, 'r-', label='Profile sent to DMD')
        ax2.set_ylabel('DMD Intensity (a.u.)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    else:
        ax1.plot(profile_x, profile_y, 'r-', label='DMD profile to load')
        ax1.set_ylabel('DMD Intensity (a.u.)')
        ax1.legend(loc='upper right')

    ax1.set_xlabel('Position (µm)')
    ax1.set_title('Profile loaded from old h5 shot')
    ax1.grid()
    plt.show()

    # 4) Send profile to DMD server
    client = zerorpc.Client()
    client.connect(f"tcp://{DMD_SERVER_IP}:{DMD_SERVER_PORT}")
    print('Server response:', client.hello())

    print('Sending old profile to DMD...')
    status = client.load_1d_profile(profile_x.tolist(), profile_y.tolist())
    print('status:', status)

    client.close()


if __name__ == '__main__':
    main()
