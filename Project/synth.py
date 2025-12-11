# synth.py
import numpy as np
from scipy.signal import lfilter

SR = 48000

def adsr_envelope(length, sr, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    n = int(length * sr)
    env = np.zeros(n, dtype=np.float32)
    a_s = int(max(1, attack * sr))
    d_s = int(max(1, decay * sr))
    r_s = int(max(1, release * sr))
    sustain_len = n - (a_s + d_s + r_s)
    if sustain_len < 0:
        # scale segments proportionally if too short
        total = attack + decay + release
        a_s = int(attack/total * n)
        d_s = int(decay/total * n)
        r_s = n - a_s - d_s
        sustain_len = 0
    # Attack
    env[:a_s] = np.linspace(0,1,a_s,endpoint=False)
    # Decay
    env[a_s:a_s+d_s] = np.linspace(1,sustain,d_s,endpoint=False)
    # Sustain
    env[a_s+d_s:a_s+d_s+sustain_len] = sustain
    # Release
    if r_s>0:
        env[-r_s:] = np.linspace(sustain,0,r_s)
    return env

def lowpass(signal, cutoff_hz, sr=SR, resonance=0.7):
    """Resonant lowpass filter (biquad approximation)."""
    if cutoff_hz >= sr/2:
        return signal
    cutoff_hz = np.clip(cutoff_hz, 20, sr/2 - 1)
    omega = 2 * np.pi * cutoff_hz / sr
    Q = np.clip(resonance, 0.1, 10.0)
    alpha = np.sin(omega) / (2 * Q)
    cos_omega = np.cos(omega)
    
    b0 = (1 - cos_omega) / 2
    b1 = 1 - cos_omega
    b2 = (1 - cos_omega) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_omega
    a2 = 1 - alpha
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1, a1/a0, a2/a0]
    return lfilter(b, a, signal)

def highpass(signal, cutoff_hz, sr=SR, resonance=0.7):
    """Resonant highpass filter."""
    if cutoff_hz <= 20:
        return signal
    cutoff_hz = np.clip(cutoff_hz, 20, sr/2 - 1)
    omega = 2 * np.pi * cutoff_hz / sr
    Q = np.clip(resonance, 0.1, 10.0)
    alpha = np.sin(omega) / (2 * Q)
    cos_omega = np.cos(omega)
    
    b0 = (1 + cos_omega) / 2
    b1 = -(1 + cos_omega)
    b2 = (1 + cos_omega) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_omega
    a2 = 1 - alpha
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1, a1/a0, a2/a0]
    return lfilter(b, a, signal)

def oscillator(wave, freq, t, sr=SR):
    """Band-limited oscillator to reduce aliasing."""
    # Handle freq as array or scalar
    freq_val = np.mean(freq) if isinstance(freq, np.ndarray) else freq
    
    if wave == 'sine':
        return np.sin(2*np.pi*freq*t)
    if wave == 'square':
        # Band-limited square wave using additive synthesis
        sig = np.zeros_like(t)
        nyquist = sr / 2
        for h in range(1, 50, 2):  # odd harmonics
            if h * freq_val < nyquist:
                sig += (1.0/h) * np.sin(2*np.pi*h*freq*t)
        return sig * 0.8
    if wave == 'saw':
        # Band-limited sawtooth using additive synthesis
        sig = np.zeros_like(t)
        nyquist = sr / 2
        for h in range(1, 50):  # all harmonics
            if h * freq_val < nyquist:
                sig += (1.0/h) * np.sin(2*np.pi*h*freq*t)
        return sig * 0.6
    if wave == 'triangle':
        # Band-limited triangle wave
        sig = np.zeros_like(t)
        nyquist = sr / 2
        for h in range(1, 50, 2):  # odd harmonics
            if h * freq_val < nyquist:
                sig += (1.0/(h*h)) * np.sin(2*np.pi*h*freq*t) * ((-1)**((h-1)//2))
        return sig * 1.2
    # fallback
    return np.sin(2*np.pi*freq*t)

def synth(params, dur=2.0, sr=SR, seed=None):
    """
    Enhanced synthesizer with FM, ring modulation, better filters.
    params: dict with keys:
      base_freq (50-2000), osc2_freq_ratio (0.5-4), osc_mix (0-1),
      osc1_wave ('sine'|'saw'|'square'|'triangle'), osc2_wave,
      lfo_rate (0-10 Hz), lfo_amt (0-0.5) -> pitch modulation,
      noise_amp (0-1), env_attack, env_decay, env_sustain, env_release,
      lp_cutoff (200-20000), hp_cutoff (20-500), filter_res (0.1-5.0),
      fm_amount (0-10), ring_mod (0-1), harmonic_spread (0-1),
      gain (0-1)
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    
    # Oscillator frequencies
    f1 = float(params.get('base_freq', 200.0))
    f2 = float(f1 * params.get('osc2_freq_ratio', 1.5))
    
    # LFO for pitch and filter modulation
    lfo_rate = params.get('lfo_rate', 5.0)
    lfo_amt = params.get('lfo_amt', 0.0)
    lfo = lfo_amt * np.sin(2*np.pi * lfo_rate * t)
    
    # FM synthesis
    fm_amount = params.get('fm_amount', 0.0)
    fm_carrier = np.sin(2*np.pi * f1 * t)
    if fm_amount > 0:
        modulator = np.sin(2*np.pi * f2 * t)
        o1 = oscillator(params.get('osc1_wave','sine'), f1*(1 + lfo + fm_amount * modulator), t, sr)
    else:
        o1 = oscillator(params.get('osc1_wave','sine'), f1*(1 + lfo), t, sr)
    
    o2 = oscillator(params.get('osc2_wave','saw'), f2*(1 + lfo), t, sr)
    
    # Oscillator mixing
    osc_mix = params.get('osc_mix', 0.6)
    sig = osc_mix*o1 + (1-osc_mix)*o2
    
    # Ring modulation
    ring_mod = params.get('ring_mod', 0.0)
    if ring_mod > 0:
        ring_sig = o1 * o2
        sig = (1 - ring_mod) * sig + ring_mod * ring_sig
    
    # Harmonic spread/richness
    harmonic_spread = params.get('harmonic_spread', 0.0)
    if harmonic_spread > 0:
        for h in [2, 3, 5]:
            if h * f1 < sr/2:
                harmonic = np.sin(2*np.pi * h * f1 * t) * (harmonic_spread / h)
                sig += harmonic
    
    # Noise shaping
    noise_amp = params.get('noise_amp', 0.01)
    if noise_amp > 0:
        noise = np.random.randn(len(t)) * noise_amp
        # Color the noise slightly
        noise_filt_cutoff = params.get('base_freq', 200.0) * 4
        noise = lowpass(noise, noise_filt_cutoff, sr, resonance=0.5)
        sig = sig + noise
    
    # Envelope
    env = adsr_envelope(dur, sr,
                        attack=params.get('env_attack', 0.01),
                        decay=params.get('env_decay', 0.1),
                        sustain=params.get('env_sustain', 0.7),
                        release=params.get('env_release', 0.1))
    sig = sig * env
    
    # Dynamic filter with envelope following
    filter_res = params.get('filter_res', 0.7)
    
    # Highpass to remove DC and rumble
    hp_cutoff = params.get('hp_cutoff', 20.0)
    if hp_cutoff > 20:
        sig = highpass(sig, hp_cutoff, sr, resonance=0.5)
    
    # Lowpass with resonance
    lp_cutoff = params.get('lp_cutoff', 8000.0)
    if lp_cutoff < sr/2:
        # Add envelope modulation to filter
        cutoff_mod = lp_cutoff * (1 + 0.5 * env * lfo_amt)
        cutoff_mod = np.clip(cutoff_mod, 100, sr/2 - 100)
        # Apply filter (use mean cutoff for stability)
        sig = lowpass(sig, np.mean(cutoff_mod), sr, resonance=filter_res)
    
    # Soft clipping for warmth
    sig = np.tanh(sig * 1.2) * 0.9
    
    # Normalize & apply gain
    g = params.get('gain', 0.8)
    maxv = np.max(np.abs(sig)) + 1e-9
    sig = (sig / maxv) * g
    
    return sig.astype(np.float32)
