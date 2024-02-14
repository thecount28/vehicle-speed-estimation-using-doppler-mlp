import numpy as np

def generate_doppler_signal(speed, frequency, time):
    wavelength = 3e8 / frequency
    doppler_shift = 2 * speed / wavelength
    doppler_signal = np.cos(2 * np.pi * (frequency + doppler_shift) * time)
    return doppler_signal

def add_noise(signal, snr_dB):
    signal_power = np.sum(np.abs(signal) ** 2) / len(signal)
    noise_power = signal_power / (10 ** (snr_dB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise
    return noisy_signal

def generate_dataset(num_samples, frequency, snr_dB, duration, sampling_rate):
    speeds = np.random.uniform(0, 30, num_samples)
    time = np.linspace(0, duration, int(duration * sampling_rate))
    X = np.zeros((num_samples, len(time)))
    y = np.zeros(num_samples)
    for i, speed in enumerate(speeds):
        doppler_signal = generate_doppler_signal(speed, frequency, time)
        noisy_signal = add_noise(doppler_signal, snr_dB)
        X[i, :] = noisy_signal
        y[i] = speed
    return X, y
