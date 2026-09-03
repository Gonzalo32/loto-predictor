import numpy as np

# ---------------------------------------------------------------------------
# Feature Engineering helpers – imported by backtest_ml_completo.py
# ---------------------------------------------------------------------------

def freq_last_k(datos_hist, k=20, max_num=45):
    """Return a dict mapping each number to its frequency in the last k draws.
    If there are fewer than k draws, use all available draws.
    """
    recent = datos_hist[-k:] if len(datos_hist) >= k else datos_hist
    freq = {n: 0.0 for n in range(max_num + 1)}
    for draw in recent:
        for n in draw:
            freq[n] += 1
    # Normalize to proportion
    total = len(recent) * 6  # 6 numbers per draw
    for n in freq:
        freq[n] = freq[n] / total if total > 0 else 0.0
    return freq


def fft_energy(series):
    """Compute total energy (sum of squared magnitudes) of the FFT of a binary series."""
    if len(series) == 0:
        return 0.0
    fft_vals = np.fft.fft(series - np.mean(series))
    energy = np.sum(np.abs(fft_vals) ** 2)
    return float(energy)

# ---------------------------------------------------------------------------
# Helper to detect abrupt distribution changes (breakpoint) – boolean flag
# ---------------------------------------------------------------------------

def breakpoint_flag(datos_hist, window=30, max_num=45, threshold=0.4):
    """Detect if the frequency distribution of numbers has changed abruptly.
    Compute the L1 distance between frequency vectors of the last *window* draws
    and the preceding *window* draws. If the distance exceeds *threshold*, return
    1, else 0.
    """
    if len(datos_hist) < 2 * window:
        return 0
    recent = datos_hist[-window:]
    older = datos_hist[-2 * window:-window]
    freq_recent = freq_last_k(recent, k=window, max_num=max_num)
    freq_older = freq_last_k(older, k=window, max_num=max_num)
    distance = sum(abs(freq_recent[n] - freq_older[n]) for n in range(max_num + 1))
    return 1 if distance > threshold else 0


def delay_entropy(delays, max_bins=5):
    """Compute the Shannon entropy of the historical delays of a number to measure predictability.
    A low entropy indicates regular recurrence intervals (high predictability).
    """
    if len(delays) < 3:
        return 0.0
    # Bin the delays into discrete intervals
    counts, _ = np.histogram(delays, bins=max_bins)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def ewma_crossover(series, short_span=3, long_span=20):
    """Return 1 if the short-term EWMA is greater than the long-term EWMA, else 0."""
    if len(series) < long_span:
        return 0.0
        
    def ewma(data, span):
        alpha = 2 / (span + 1)
        val = data[0]
        for x in data[1:]:
            val = alpha * x + (1 - alpha) * val
        return val
        
    short_val = ewma(series, short_span)
    long_val = ewma(series, long_span)
    return 1.0 if short_val > long_val else 0.0



