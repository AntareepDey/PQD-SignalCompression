import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from sklearn.linear_model import OrthogonalMatchingPursuit
import pywt
from collections import Counter
import heapq

def generate_impulse_signal(fs=6400, duration=0.1, snr=40):
    t = np.arange(0, duration, 1/fs)
    impulse_center = int(0.06 * fs)
    impulse_width = int(0.001 * fs)
    signal = np.sin(2 * np.pi * 50 * t)
    signal[impulse_center:impulse_center + impulse_width] += np.exp(-np.arange(impulse_width))
    return add_noise(signal, snr)

def generate_sag_signal(fs=6400, duration=0.1, snr=40):
    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2 * np.pi * 50 * t)
    sag_start, sag_end = int(0.04 * fs), int(0.08 * fs)
    signal[sag_start:sag_end] *= 0.5
    return add_noise(signal, snr)

def generate_decaying_harmonics_signal(fs=6400, duration=0.1, snr=40):
    t = np.arange(0, duration, 1/fs)
    signal = np.exp(-2 * np.pi * t) * (
        np.sin(2 * np.pi * 50 * t) +
        0.5 * np.sin(2 * np.pi * 150 * t + np.pi / 4) +
        0.3 * np.sin(2 * np.pi * 250 * t + np.pi / 3)
    )
    return add_noise(signal, snr)

def add_noise(signal, snr):
    power_signal = np.mean(signal ** 2)
    power_noise = power_signal / (10 ** (snr / 10))
    noise = np.sqrt(power_noise) * np.random.randn(len(signal))
    return signal + noise

# --- Step 2: Sparse Decomposition ---
def create_overcomplete_dictionary(n_samples):
    identity_matrix = np.eye(n_samples)
    hartley_matrix = hartley_transform_matrix(n_samples)
    return np.hstack((identity_matrix, hartley_matrix))

def hartley_transform_matrix(n_samples):
    indices = np.arange(n_samples)
    matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        matrix[i, :] = np.cos(2 * np.pi * i * indices / n_samples) + np.sin(2 * np.pi * i * indices / n_samples)
    return matrix / np.sqrt(n_samples)

def sparse_decomposition_omp(signal, dictionary):
    signal = signal.reshape(-1, 1)  # Reshape to (n_samples, 1) for OMP
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=50, tol=1e-3)
    omp.fit(dictionary, signal)
    sparse_code = omp.coef_.flatten()
    return sparse_code

# --- Step 3: Strong Tracking Kalman Filter for Wavelet Threshold ---
def strong_tracking_kalman_filter(signal, fs):
    n = len(signal)
    FF = np.ones(n)
    # State vector: [amplitude, frequency, phase]
    x_k = np.zeros((3, 1))  # Reshape to column vector
    F = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    H = np.array([[1, 0, 0]])  # Measurement matrix (1x3)
    Q = 1e-5 * np.eye(3)
    R = 1e-4
    P = np.eye(3)

    for k in range(1, n):
        # Prediction
        x_k = F @ x_k
        P = F @ P @ F.T + Q
        
        # Update
        K = P @ H.T / (H @ P @ H.T + R)  # Kalman gain (3x1)
        residual = signal[k] - (H @ x_k)[0, 0]  # Scalar residual
        x_k = x_k + K * residual  # Update state
        P = P - (K @ H @ P)  # Update covariance
        
        # Strong tracking factor
        FF[k] = np.sqrt(residual**2 / (H @ P @ H.T + R))
    
    return FF

def wavelet_threshold_with_stkf(signal, wavelet="db4", level=3, fs=6400):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Get strong tracking factors
    FF = strong_tracking_kalman_filter(signal, fs)
    mean_FF = np.mean(FF)
    
    # Apply adaptive thresholding
    thresholded_coeffs = []
    for i, c in enumerate(coeffs):
        if i == 0:  # Approximation coefficients
            thresholded_coeffs.append(c)
        else:
            # Adaptive threshold based on strong tracking factor
            α = 2 if mean_FF > 1 else 1
            threshold = (np.median(np.abs(c)) / 0.6475) * α * np.sqrt(2 * np.log(len(signal)))
            thresholded_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
    
    return thresholded_coeffs

# --- Signal Generation ---
fs = 6400
duration = 0.1
snr = 40

signals = {
    "Impulse": generate_impulse_signal(fs, duration, snr),
    "Sag": generate_sag_signal(fs, duration, snr),
    "Decaying Harmonics": generate_decaying_harmonics_signal(fs, duration, snr)
}

# --- Apply Sparse Decomposition and Thresholding ---
results = {}
for name, signal in signals.items():
    n_samples = len(signal)
    dictionary = create_overcomplete_dictionary(n_samples)
    sparse_code = sparse_decomposition_omp(signal, dictionary)
    compressed_coeffs = wavelet_threshold_with_stkf(sparse_code[:n_samples])
    results[name] = {"original": signal, "compressed": compressed_coeffs}

# --- Visualization ---
for name, result in results.items():
    plt.figure(figsize=(12, 6))
    plt.plot(result["original"], label=f"Original {name} Signal")
    plt.plot(pywt.waverec(result["compressed"], wavelet="db4"), label=f"Reconstructed {name} Signal", linestyle="dashed")
    plt.title(f"{name} Signal: Original vs Reconstructed")
    plt.legend()
    plt.grid()
    plt.show()