from collections import namedtuple

DataSource = namedtuple('DataSource', "name n_series, n_timesteps")

MNIST_DS = DataSource(name="mnist",
                      n_series=1,
                      n_timesteps=28)

MNIST_RESHAPED_DS = DataSource(name="mnist_reshaped",
                               n_series=28,
                               n_timesteps=28)

LOW_NOISE_DS = DataSource(name="low_noise",
                          n_series=1,
                          n_timesteps=100)


