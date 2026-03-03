import os, h5py, glob
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

# Worker function must live at module scope so it can be pickled by ProcessPoolExecutor.
def _worker_write_shared(args):
    """
    Worker executed in a separate process. Reads one snapshot from disk and writes
    it into the provided shared-memory buffers at the destination index.

    Args (tuple):
        snap_index (int)
        dest_idx (int)
        file_path (str)
        shm_dark_name (str or None)
        shape_dark (tuple)  # (num_snapshots, N_dark, 6) -- ALWAYS passed
        dtype_name (str)
        shm_star_name (str or None)
        shape_star (tuple or None)  # (num_snapshots, N_star, 6) or None
    Returns:
        snap_index (int)
    """
    (
        snap_index,
        dest_idx,
        file_path,
        shm_dark_name,
        shape_dark,
        dtype_name,
        shm_star_name,
        shape_star,
    ) = args

    import numpy as _np
    from multiprocessing import shared_memory as _shared_memory
    import h5py as _h5py
    
    dtype = _np.dtype(dtype_name)
    dset = f"snap.{snap_index:03d}"

    # Read snapshot (each worker opens its own file handle)
    with _h5py.File(file_path, "r") as f:
        raw = f["snapshots"][dset][:]

    # Number of dark / star particles (from shapes passed)
    # shape_dark is always provided by the parent, even if shm_dark is None.
    num_dark = int(shape_dark[1])
    num_star = int(shape_star[1]) if shape_star is not None else None

    # Write into dark shared memory if provided
    if shm_dark_name is not None:
        shm_d = _shared_memory.SharedMemory(name=shm_dark_name)
        dark_view = _np.ndarray(shape_dark, dtype=dtype, buffer=shm_d.buf)
        # raw[0:num_dark] -> (N_dark,6)
        dark_view[dest_idx, :, :] = raw[0:num_dark]
        shm_d.close()

    # Write into star shared memory if provided
    if shm_star_name is not None and shape_star is not None:
        shm_s = _shared_memory.SharedMemory(name=shm_star_name)
        star_view = _np.ndarray(shape_star, dtype=dtype, buffer=shm_s.buf)
        # raw indexing: stars start after dark block
        start = num_dark
        end = start + num_star
        star_view[dest_idx, :, :] = raw[start:end]
        shm_s.close()

    return snap_index

class ParticleReader:
    """
    A class to read N-body simulation data from one or more HDF5 files.

    This reader assumes a consistent data structure across files, with particle
    data stored in a '/snapshots' group where each dataset is named 'snap.###'.

    Units assumed:
      - Length: kpc
      - Mass: Msun
      - Velocity: km/s
      - Time: physical time in Gyr (if a snapshot.times file is provided)

    Usage
    ----------
    Simreader = ParticleReader(sim_dir, verbose=True)
    part = Simreader.read_snapshot(100) # # load in snap=100, also accepts float as time.
    orbits = Simreader.extract_orbits('star', 4)    

    Parameters
    ----------
    sim_pattern : str
        A path or glob pattern for the HDF5 files (e.g., 'path/to/sim.hdf5'
        or 'path/to/sim.*.hdf5').
    times_file_path : str or Path, optional
        Path to the snapshot times file. If not provided, the reader will look
        for a file named 'snapshot.times' in the parent directory of the
        first matched simulation file (i.e., `Path(first_file).parent / 'snapshot.times'`).
        If that file does not exist it will be set to None.
    verbose : bool, optional
        If True, print status messages. Default is False.
    """

    def __init__(self, sim_pattern: str, times_file_path: str = None, verbose: bool = False):
        self._verbose = bool(verbose)
        if self._verbose:
            print(f"🔍 Initializing ParticleReader for pattern: {sim_pattern}")

        self.file_list = sorted(glob.glob(sim_pattern))
        if not self.file_list:
            raise FileNotFoundError(f"❌ No HDF5 files found matching pattern: {sim_pattern}")
        if self._verbose:
            print(f"   Found {len(self.file_list)} simulation file(s).")

        # read properties and map snapshots before trying to locate times file
        self._read_properties()
        self._map_snapshots_to_files()

        # Determine default times file path based on sim_pattern location.
        # Default: <first_matched_file>.parent / 'snapshot.times'
        # If that file doesn't exist we set times_file_path = None.
        if times_file_path is None:
            default_path = Path(self.file_list[0]).parent / "snapshot.times"
            if default_path.exists():
                times_file_path = default_path
                if self._verbose:
                    print(f"   Using default times file: {default_path}")
            else:
                times_file_path = None
                if self._verbose:
                    print(f"⚠️   snapshot.times not found at expected location: {default_path} -> Times disabled")

        self.Times = None
        if times_file_path is not None:
            try:
                # np.loadtxt accepts Path-like in newer numpy, but cast to str for safety.
                self.Times = np.loadtxt(str(times_file_path), comments="#")
                if self._verbose:
                    print(f"✅  Successfully loaded time mapping from: {times_file_path}")
            except OSError:
                # file missing or unreadable
                if self._verbose:
                    print(f"⚠️   snapshot.times file not found or unreadable at: {times_file_path}. Times disabled.")
                self.Times = None
            except Exception as e:
                if self._verbose:
                    print(f"⚠️   Warning: Failed to load time mapping ({e}). Times disabled.")
                self.Times = None

    def _read_properties(self):
        """
        Read simulation properties from the first HDF5 file.
        These properties are assumed constant across files.
        """
        with h5py.File(self.file_list[0], "r") as f:
            self.num_dark = int(f["/properties/dark/N"][()])
            self.num_star = int(f["/properties/star/N"][()])
            # masses (Msun)
            self.mass_dark = float(f["/properties/dark/m"][()])
            self.mass_star = float(f["/properties/star/m"][()])

            # timestep in time units
            self._timestep = float(f["/properties/time_step"][()])

            # softening lengths
            self._eps_dark = float(f["/properties/dark/eps"][()])
            try:
                self._eps_star = float(f["/properties/star/eps"][()])
            except Exception:
                if self._verbose:
                    print("⚠️ star eps missing. Assuming dark eps.")
                self._eps_star = float(f["/properties/dark/eps"][()])

        if self._verbose:
            print("✅   Simulation properties:")
            print(f"      - N_dark: {self.num_dark}, N_star: {self.num_star}")
            print(f"      - M_dark: {self.mass_dark:.2e}, M_star: {self.mass_star:.2e}")

    def _map_snapshots_to_files(self):
        """
        Scan HDF5 files and map snapshot index -> file path for fast lookups.

        Sets:
          - self._snap_to_file_map : dict mapping snap_index -> file_path
          - self.Snapshots : np.ndarray sorted snapshot indices (dtype=int)
        """
        self._snap_to_file_map = {}
        if self._verbose:
            print("   Scanning files to map snapshot locations...")
        for file_path in self.file_list:
            with h5py.File(file_path, "r") as f:
                if "snapshots" not in f:
                    continue
                for snap_name in f["snapshots"].keys():
                    # expected snap_name like 'snap.012' or 'snap.12'
                    try:
                        snap_index = int(snap_name.split(".")[-1])
                    except ValueError:
                        # skip non-standard snapshot names
                        continue
                    self._snap_to_file_map[snap_index] = file_path

        # Create a sorted numpy array of snapshot indices for easy access
        if len(self._snap_to_file_map) > 0:
            self.Snapshots = np.array(sorted(self._snap_to_file_map.keys()), dtype=int)
        else:
            self.Snapshots = np.array([], dtype=int)

        if self._verbose:
            print(f"   ...found {len(self._snap_to_file_map)} total snapshots.")

    def read_snapshot(self, identifier: int | float):
        """
        Read a single snapshot by index or physical time.

        Parameters
        ----------
        identifier : int or float
            If int: snapshot index (e.g., 151).
            If float: physical time in Gyr (e.g., 7.5) — requires snapshot.times file.

        Returns
        -------
        types.SimpleNamespace
            - part.dark / part.star: dicts containing 'posvel' (N x 6) and 'mass' arrays.
            - part.snap: int, snapshot index.
            - part.time: float or None, physical time in Gyr if available.
        """
        if isinstance(identifier, float):
            if self.Times is None:
                raise ValueError("❌ Time-based lookup requires a snapshot.times file, which was not loaded.")
            time_col = self.Times[:, 1]
            closest_time_idx = np.argmin(np.abs(time_col - identifier))
            snap_index = int(self.Times[closest_time_idx, 0])
        elif isinstance(identifier, int):
            snap_index = identifier
        else:
            raise TypeError("❌ Identifier must be an integer (snapshot index) or a float (time).")

        if snap_index not in self._snap_to_file_map:
            raise ValueError(f"❌ Snapshot index {snap_index} not found in any of the simulation files.")

        file_to_open = self._snap_to_file_map[snap_index]
        dataset_name = f"snap.{snap_index:03d}"

        with h5py.File(file_to_open, "r") as f:
            raw_data = f["snapshots"][dataset_name][:]

        part = SimpleNamespace(dark={}, star={})

        dm_end_index = self.num_dark
        part.dark["posvel"] = raw_data[0:dm_end_index]
        part.star["posvel"] = raw_data[dm_end_index : dm_end_index + self.num_star]

        part.dark["mass"] = np.full(self.num_dark, self.mass_dark)
        part.star["mass"] = np.full(self.num_star, self.mass_star)

        part.snap = snap_index
        if self.Times is not None:
            time_row = self.Times[self.Times[:, 0] == snap_index]
            part.time = float(time_row[0, 1]) if len(time_row) > 0 else None
        else:
            part.time = None

        if self._verbose:
            print(f"✅   Read snapshot {snap_index} from {file_to_open} (time={part.time})")

        return part

    def extract_orbits(self, particle_type: str = "star", min_parallel_workers: int = 4):
        """
        Extract orbits for selected particle types across all available snapshots.
        using parallel worker processes that write directly into shared memory.

        WARNING: This loads *all* snapshot phase-space data for the chosen particle
        types into memory. It can be very memory intensive for many snapshots and
        large N.

        Parameters
        ----------
        particle_type : {'dark', 'star', 'all', True} or False
            Which particle types to load. Use 'dark', 'star', or 'all' (or True).
            If False, the function returns None.
        min_parallel_workers : Number of parallel workers. Default = 4. 
            Set to min(min_parallel_workers, CPUsavail, num_snapshots)
            
        Returns
        -------
        types.SimpleNamespace
            - orbits.dark / orbits.star : ndarray with shape (num_snapshots, N_particles, 6)
              (snapshot-major ordering)
            - orbits.Times : 1D ndarray of length num_snapshots with physical times (Gyr),
              or None if no times were loaded.
            - orbits.Snaps : 1D ndarray of snapshot indices (present only if Times not available).
        """

        if not particle_type:
            return None

        types_to_process = []
        if particle_type in ["dark", "all", True]:
            types_to_process.append("dark")
        if particle_type in ["star", "all", True]:
            types_to_process.append("star")

        if not types_to_process:
            raise ValueError(f"Invalid particle_type: {particle_type}. Use 'dark', 'star', or 'all'.")

        all_snap_indices = sorted(self._snap_to_file_map.keys())
        num_snapshots = len(all_snap_indices)

        orbits = SimpleNamespace()

        if num_snapshots == 0:
            if self._verbose:
                print("⚠️ No snapshots found; returning empty orbits.")
            if "dark" in types_to_process:
                orbits.dark = np.zeros((0, self.num_dark, 6), dtype=np.float64)
            if "star" in types_to_process:
                orbits.star = np.zeros((0, self.num_star, 6), dtype=np.float64)
            orbits.Times = None
            orbits.Snaps = np.array([], dtype=int)
            return orbits

        # Shared memory shapes: (num_snapshots, N_particles, 6)
        shape_dark = (num_snapshots, self.num_dark, 6)
        shape_star = (num_snapshots, self.num_star, 6) if "star" in types_to_process else None

        dtype = np.float64
        dtype_name = np.dtype(dtype).name

        shm_dark = None
        shm_star = None
        shared_dark = None
        shared_star = None

        def _nbytes_for_shape(s, dt):
            return int(np.prod(s)) * int(np.dtype(dt).itemsize)

        try:
            # Create shared memory buffers (only when the corresponding type is requested)
            if "dark" in types_to_process:
                bytes_dark = _nbytes_for_shape(shape_dark, dtype)
                shm_dark = shared_memory.SharedMemory(create=True, size=bytes_dark)
                shared_dark = np.ndarray(shape_dark, dtype=dtype, buffer=shm_dark.buf)
                shared_dark[:] = 0.0

            if "star" in types_to_process:
                bytes_star = _nbytes_for_shape(shape_star, dtype)
                shm_star = shared_memory.SharedMemory(create=True, size=bytes_star)
                shared_star = np.ndarray(shape_star, dtype=dtype, buffer=shm_star.buf)
                shared_star[:] = 0.0

            # Build argument list for each snapshot (pass shape_dark always so worker knows num_dark)
            args_list = []
            for dest_idx, snap in enumerate(all_snap_indices):
                file_path = self._snap_to_file_map[snap]
                args = (
                    int(snap),
                    int(dest_idx),
                    str(file_path),
                    shm_dark.name if shm_dark is not None else None,
                    shape_dark,           # ALWAYS pass shape_dark
                    dtype_name,
                    shm_star.name if shm_star is not None else None,
                    shape_star,           # pass shape_star or None
                )
                args_list.append(args)

            cpu_avail = os.cpu_count() or 1
            max_workers = min(min_parallel_workers, cpu_avail, num_snapshots)

            if self._verbose:
                print(f"   Spawning up to {max_workers} workers to read {num_snapshots} snapshots...")

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_worker_write_shared, a): a[0] for a in args_list}
                for fut in as_completed(futures):
                    # ensure exceptions are raised here for visibility
                    _ = fut.result()

            # Copy shared memory views into normal numpy arrays for return
            if "dark" in types_to_process:
                orbits.dark = np.array(shared_dark, copy=True)
            if "star" in types_to_process:
                orbits.star = np.array(shared_star, copy=True)

        finally:
            # Cleanup shared memory
            if shm_dark is not None:
                try:
                    shm_dark.close()
                    shm_dark.unlink()
                except FileNotFoundError:
                    pass
            if shm_star is not None:
                try:
                    shm_star.close()
                    shm_star.unlink()
                except FileNotFoundError:
                    pass

        # Attach time information (1D array aligned with snapshots) or Snaps
        if self.Times is not None:
            try:
                times_arr = np.asarray(self.Times)
                if times_arr.ndim == 1:
                    if times_arr.shape[0] == num_snapshots:
                        orbits.Times = times_arr.astype(float)
                    else:
                        orbits.Times = None
                        if self._verbose:
                            print("   Warning: Times is 1D but length doesn't match number of snapshots; omitting orbits.Times.")
                else:
                    snap_idxs = times_arr[:, 0].astype(int)
                    time_vals = times_arr[:, 1].astype(float)
                    if np.array_equal(snap_idxs, np.array(all_snap_indices, dtype=int)):
                        orbits.Times = time_vals
                    else:
                        time_map = dict(zip(snap_idxs, time_vals))
                        orbits.Times = np.array([time_map.get(snap, np.nan) for snap in all_snap_indices], dtype=float)
            except Exception:
                orbits.Times = None
                if self._verbose:
                    print("   Warning: could not parse Times array into orbits.Times.")
        else:
            orbits.Snaps = np.array(all_snap_indices, dtype=int)
            orbits.Times = None
            if self._verbose:
                print("   Times not available — attaching orbits.Snaps (snapshot indices).")

        if self._verbose:
            print("✅  ...extraction complete.")

        return orbits