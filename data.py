import numpy as np
import h5py
import netCDF4 as nc

class BatchGenerator(object):
    
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = self.data[0].shape[0]
        self.next_ind = np.array([], dtype=int)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.next_ind) < self.batch_size:
            ind = np.arange(self.N, dtype=int)
            print('ind len', len(ind))
            if self.shuffle:
                np.random.shuffle(ind)
            self.next_ind = np.concatenate([self.next_ind, ind])

        ind = self.next_ind[:self.batch_size]
        self.next_ind = self.next_ind[self.batch_size:]

        x_batch = self.data[0][ind,...]
        y_batch = self.data[1][ind,...]
        mask_batch = self.data[2][ind,...]

        return (x_batch,y_batch,mask_batch)

    
class SSHBatchGenerator(BatchGenerator):
    def __init__(self, file_path, **kwargs):
        # Load and prepare data
        self.file_path = file_path
        data = self.load_data()
        super(SSHBatchGenerator, self).__init__(data, **kwargs)

    def load_data(self):
        # Load data from the netCDF file
        ds = nc.Dataset(self.file_path)
        ntrain = 1000  # Number of training days (train + valid)
        train_end_idx = ds.variables['patch_start_indices'][ntrain]

        # Load SSH and SLA data (both 2D arrays) with a shape [time, height, width]
        ssha = ds.variables['ssha'][:, :, :]  # [time, lat, lon]
        sla = ds.variables['sla'][:, :, :]   # [time, lat, lon]
        mask = 1 - ds.variables['ssha_mask'][:, :, :]

        target_lat = ssha.shape[1]
        target_lon = ssha.shape[2]
        cond_lat = sla.shape[1]
        cond_lon = sla.shape[2]
        ds.close()
        # Reshaping into the form expected by the model (e.g., [samples, height, width, channels])
        X_train = ssha[:train_end_idx, :, :].reshape((-1, target_lat, target_lon, 1))  # Targets
        y_train = sla[:train_end_idx, :, :].reshape((-1, cond_lat, cond_lon, 1))  # Conditioning
        mask_train = mask[:train_end_idx, :, :].reshape((-1, target_lat, target_lon, 1))  # mask

        # Return data as a tuple
        return (X_train, y_train, mask_train)

class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=32, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __iter__(self):
        return self

    def __next__(self, mean=0.0, std=1.0):

        def noise(shape):
            shape = (self.batch_size,) + shape

            n = self.prng.randn(*shape).astype(np.float32)
            if std != 1.0:
                n *= std
            if mean != 0.0:
                n += mean
            return n

        return [noise(s) for s in self.noise_shapes]