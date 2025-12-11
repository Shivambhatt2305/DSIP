# Creative Text-to-Audio Generation via Synthesizer Programming

An AI-powered system that generates audio from text descriptions by optimizing synthesizer parameters using CLAP (Contrastive Language-Audio Pretraining) embeddings.

## Features

### Enhanced Synthesizer Engine
- **Band-Limited Oscillators**: Sine, sawtooth, square, and triangle waveforms with reduced aliasing
- **FM Synthesis**: Frequency modulation for complex timbres
- **Ring Modulation**: Creates metallic and inharmonic sounds
- **Dual Oscillators**: Mix between two independent oscillators with different waveforms
- **Resonant Filters**: Both lowpass and highpass filters with resonance control
- **ADSR Envelope**: Attack, Decay, Sustain, Release envelope shaping
- **LFO Modulation**: Low-frequency oscillator for pitch and filter modulation
- **Harmonic Spread**: Add harmonic richness to sounds
- **Noise Generator**: Shaped noise with frequency-dependent filtering

### Intelligent Optimization
- **CLAP-Based Matching**: Uses state-of-the-art audio-text embeddings
- **Adaptive Mutation**: Dynamically adjusts search strategy based on progress
- **Elite Selection**: Preserves best candidates across generations
- **Diversity Injection**: Prevents premature convergence to local optima
- **Cosine Similarity**: Accurate distance metric between audio and text embeddings

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --prompt "bees buzzing" --out bees.wav
```

### Advanced Options
```bash
python main.py --prompt "ocean waves crashing" \
    --out ocean.wav \
    --gens 200 \
    --pop 40 \
    --dur 3.0
```

### Parameters
- `--prompt`: Text description of the sound to generate (required)
- `--out`: Output filename (optional, auto-generated if not specified)
- `--gens`: Number of optimization generations (default: 150, more = better quality)
- `--pop`: Population size for genetic algorithm (default: 30)
- `--dur`: Duration of generated sound in seconds (default: 2.0)
- `--outdir`: Output directory (default: 'outputs')

## How It Works

1. **Text Embedding**: Convert text prompt to embedding using CLAP model
2. **Parameter Initialization**: Create random synthesizer parameter populations
3. **Synthesis**: Generate audio using band-limited synthesis with FM, filters, etc.
4. **Audio Embedding**: Convert generated audio to embedding
5. **Fitness Evaluation**: Calculate cosine similarity between text and audio embeddings
6. **Evolution**: Use adaptive genetic algorithm to optimize parameters
7. **Output**: Save best-matching audio and parameters

## Synthesizer Parameters

The system optimizes 20+ parameters:
- **Oscillators**: base_freq, osc2_freq_ratio, osc_mix, waveforms
- **Modulation**: lfo_rate, lfo_amt, fm_amount, ring_mod
- **Envelope**: env_attack, env_decay, env_sustain, env_release
- **Filters**: lp_cutoff, hp_cutoff, filter_res
- **Texture**: noise_amp, harmonic_spread
- **Output**: gain

## Output Files

For each generation, the system saves:
- `*.wav`: Generated audio file
- `*_spec.png`: Spectrogram visualization
- `*_params.txt`: Optimized synthesizer parameters

## Examples

```bash
# Natural sounds
python main.py --prompt "wind howling" --out wind.wav
python main.py --prompt "rain on metal roof" --out rain.wav
python main.py --prompt "bird chirping" --out bird.wav

# Mechanical sounds
python main.py --prompt "helicopter rotor" --out helicopter.wav
python main.py --prompt "car engine idling" --out engine.wav

# Abstract sounds
python main.py --prompt "mysterious ambience" --out mystery.wav
python main.py --prompt "digital glitch" --out glitch.wav
```

## Performance Tips

- **Quick test**: Use `--gens 50 --pop 20` for faster results
- **High quality**: Use `--gens 250 --pop 40` for best matching
- **Longer sounds**: Use `--dur 4.0` or higher for evolving textures

## Technical Details

- **Sample Rate**: 48kHz
- **Synthesis**: Real-time additive/subtractive with FM
- **Embedding Model**: LAION CLAP (512-dim vectors)
- **Optimization**: Adaptive evolutionary algorithm with elitism
- **Audio Features**: MFCC + spectral features (fallback mode)

## Requirements

- Python 3.8+
- PyTorch
- Transformers (for CLAP)
- librosa
- scipy
- numpy
- soundfile
- matplotlib

## Troubleshooting

**Low similarity scores**: Increase `--gens` and `--pop` for better optimization

**Sound quality issues**: Check that CLAP model loaded successfully (see console output)

**Memory errors**: Reduce `--pop` size or use shorter `--dur`

## Future Enhancements

- Multi-objective optimization (similarity + diversity)
- Wavetable synthesis
- Effects processing (reverb, delay, distortion)
- Real-time parameter control
- Batch generation mode

## License

MIT License
