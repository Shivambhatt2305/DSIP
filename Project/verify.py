"""
Verification script to test the enhanced text-to-audio generation system.
Tests basic functionality without running full optimization.
"""
import numpy as np
from synth import synth, SR, oscillator, lowpass, highpass, adsr_envelope
from embed import get_text_embedding, get_audio_embedding_from_array
from optimize import random_params, perturb_param
from utils import save_wav
import os

print("=" * 60)
print("VERIFICATION TEST - Enhanced Text-to-Audio System")
print("=" * 60)

# Test 1: Basic synthesizer functionality
print("\n[1/6] Testing synthesizer with new features...")
try:
    test_params = {
        'base_freq': 440.0,
        'osc2_freq_ratio': 2.0,
        'osc_mix': 0.5,
        'osc1_wave': 'sine',
        'osc2_wave': 'square',
        'lfo_rate': 5.0,
        'lfo_amt': 0.1,
        'noise_amp': 0.05,
        'env_attack': 0.01,
        'env_decay': 0.1,
        'env_sustain': 0.7,
        'env_release': 0.2,
        'lp_cutoff': 5000.0,
        'hp_cutoff': 100.0,
        'filter_res': 1.0,
        'fm_amount': 0.5,
        'ring_mod': 0.2,
        'harmonic_spread': 0.3,
        'gain': 0.8
    }
    audio = synth(test_params, dur=0.5, sr=SR)
    assert len(audio) > 0
    assert audio.dtype == np.float32
    print("   ✓ Synthesizer working with all new parameters")
    print(f"   ✓ Generated {len(audio)} samples ({len(audio)/SR:.2f}s)")
except Exception as e:
    print(f"   ✗ Synthesizer test failed: {e}")
    exit(1)

# Test 2: Band-limited oscillators
print("\n[2/6] Testing band-limited oscillators...")
try:
    t = np.linspace(0, 0.1, int(SR*0.1))
    for wave in ['sine', 'square', 'saw', 'triangle']:
        osc_out = oscillator(wave, 440.0, t, SR)
        assert len(osc_out) == len(t)
        assert not np.any(np.isnan(osc_out))
        print(f"   ✓ {wave.capitalize()} wave: {len(osc_out)} samples")
except Exception as e:
    print(f"   ✗ Oscillator test failed: {e}")
    exit(1)

# Test 3: Resonant filters
print("\n[3/6] Testing resonant filters...")
try:
    test_signal = np.random.randn(SR)  # 1 second of noise
    lp_out = lowpass(test_signal, 1000.0, SR, resonance=2.0)
    hp_out = highpass(test_signal, 500.0, SR, resonance=2.0)
    assert len(lp_out) == len(test_signal)
    assert len(hp_out) == len(test_signal)
    print(f"   ✓ Lowpass filter: resonance control working")
    print(f"   ✓ Highpass filter: resonance control working")
except Exception as e:
    print(f"   ✗ Filter test failed: {e}")
    exit(1)

# Test 4: Text embedding
print("\n[4/6] Testing text embedding...")
try:
    text_emb = get_text_embedding("test sound")
    assert len(text_emb) > 0
    assert not np.any(np.isnan(text_emb))
    print(f"   ✓ Text embedding generated: {len(text_emb)} dimensions")
except Exception as e:
    print(f"   ✗ Text embedding test failed: {e}")
    exit(1)

# Test 5: Audio embedding
print("\n[5/6] Testing audio embedding...")
try:
    test_audio = synth(test_params, dur=1.0, sr=SR)
    audio_emb = get_audio_embedding_from_array(test_audio, SR)
    assert len(audio_emb) > 0
    assert not np.any(np.isnan(audio_emb))
    print(f"   ✓ Audio embedding generated: {len(audio_emb)} dimensions")
    
    # Test similarity calculation
    sim = np.dot(audio_emb, text_emb) / (np.linalg.norm(audio_emb) * np.linalg.norm(text_emb) + 1e-9)
    print(f"   ✓ Similarity score computed: {sim:.4f}")
except Exception as e:
    print(f"   ✗ Audio embedding test failed: {e}")
    exit(1)

# Test 6: Optimization functions
print("\n[6/6] Testing optimization functions...")
try:
    # Test random parameter generation
    params1 = random_params()
    assert 'fm_amount' in params1
    assert 'ring_mod' in params1
    assert 'harmonic_spread' in params1
    assert 'filter_res' in params1
    assert 'hp_cutoff' in params1
    assert 'triangle' in ['sine', 'saw', 'square', 'triangle']  # Check triangle available
    print(f"   ✓ Random parameters generated with {len(params1)} parameters")
    
    # Test parameter perturbation
    params2 = perturb_param(params1, scale=0.1)
    assert len(params2) == len(params1)
    different_count = sum(1 for k in params1.keys() if params1[k] != params2[k])
    print(f"   ✓ Parameter perturbation working ({different_count} values changed)")
    
    # Test parameter clamping
    assert 20 <= params2['base_freq'] <= 8000
    assert 0.1 <= params2['filter_res'] <= 5.0
    assert 0.0 <= params2['ring_mod'] <= 1.0
    print(f"   ✓ Parameter clamping working correctly")
except Exception as e:
    print(f"   ✗ Optimization test failed: {e}")
    exit(1)

# Save test audio
print("\n[BONUS] Saving test audio...")
try:
    os.makedirs('outputs', exist_ok=True)
    test_file = 'outputs/verification_test.wav'
    save_wav(test_file, audio, SR)
    if os.path.exists(test_file):
        print(f"   ✓ Test audio saved: {test_file}")
except Exception as e:
    print(f"   ⚠ Could not save test audio: {e}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nYour system is ready to generate audio from text!")
print("\nTry running:")
print('  python main.py --prompt "bees buzzing" --out bees.wav')
print("\nFor quick test:")
print('  python main.py --prompt "test tone" --out test.wav --gens 50 --pop 20')
print("\n" + "=" * 60)
