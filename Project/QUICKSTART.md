# Quick Start Guide

## Run Your First Generation

```bash
python main.py --prompt "bees buzzing" --out bees.wav
```

This will:
1. Load the CLAP model (takes ~1 minute first time)
2. Run 150 generations of optimization (~3-5 minutes)
3. Save three files in `outputs/`:
   - `bees.wav` - The generated audio
   - `spec_bees.png` - Spectrogram visualization
   - `params_bees.txt` - Synthesizer parameters used

## Command Format

```bash
python main.py --prompt "YOUR TEXT HERE" --out filename.wav [OPTIONS]
```

### Required Arguments
- `--prompt "text"` - Description of sound to generate

### Optional Arguments
- `--out filename.wav` - Specific output name (default: auto-generated)
- `--gens 150` - Number of generations (default: 150)
  - Lower (50-100): Faster, lower quality
  - Higher (200-300): Slower, better quality
- `--pop 30` - Population size (default: 30)
- `--dur 2.0` - Duration in seconds (default: 2.0)
- `--outdir outputs` - Output directory (default: outputs)

## Example Commands

### Quick Test (Fast)
```bash
python main.py --prompt "bell ringing" --out bell.wav --gens 50 --pop 20
```

### Standard Quality (Recommended)
```bash
python main.py --prompt "ocean waves" --out ocean.wav
```

### High Quality (Slow but Best)
```bash
python main.py --prompt "thunder rumble" --out thunder.wav --gens 250 --pop 40
```

### Longer Duration
```bash
python main.py --prompt "wind through trees" --out wind.wav --dur 5.0
```

## Example Prompts

### Natural Sounds
- "bees buzzing"
- "ocean waves crashing"
- "wind howling"
- "rain on metal roof"
- "bird chirping"
- "crickets at night"
- "thunder rumble"
- "campfire crackling"

### Mechanical Sounds
- "helicopter rotor"
- "car engine idling"
- "drill machine"
- "electric buzzing"
- "steam hissing"
- "clock ticking"

### Musical Elements
- "deep bass note"
- "bell ringing"
- "strings swelling"
- "electronic beep"
- "cymbal crash"

### Abstract Sounds
- "mysterious ambience"
- "digital glitch"
- "metallic resonance"
- "sci-fi laser"
- "alien communication"

## Reading the Output

### Console Output
```
[embed] CLAP model loaded.
Prompt: bees buzzing
Generations: 100%|████████| 150/150 [03:42<00:00]
[gen 0] best=0.2134 global=0.2134 no_improve=0
[gen 10] best=0.2891 global=0.3012 no_improve=2
[gen 20] best=0.3156 global=0.3421 no_improve=0
...
```

- **best**: Best similarity score this generation
- **global**: Best score found so far
- **no_improve**: Generations since last improvement (triggers adaptive search)

### Similarity Scores
- **0.0 - 0.2**: Poor match, may need more generations
- **0.2 - 0.4**: Decent match, recognizable sound
- **0.4 - 0.6**: Good match, accurate representation
- **0.6+**: Excellent match, very accurate

### Output Files

1. **Audio File (.wav)**
   - 48kHz sample rate
   - Float32 format
   - Play with any audio player

2. **Spectrogram (.png)**
   - Visual representation of frequencies over time
   - Vertical axis: Frequency (Hz)
   - Horizontal axis: Time (seconds)
   - Color: Amplitude (dB)

3. **Parameters (.txt)**
   ```
   base_freq: 234.56
   osc1_wave: saw
   osc2_wave: sine
   ...
   ```
   - All 20+ synthesizer parameters
   - Can be used to reproduce the sound
   - Useful for understanding what makes the sound

## Troubleshooting

### "CLAP model unavailable, fallback will be used"
- This is OK! Fallback mode still works
- Install transformers model: `pip install transformers --upgrade`
- May need internet connection for first download

### Low Similarity Scores (<0.3 after 150 gens)
- Try more generations: `--gens 250`
- Try larger population: `--pop 40`
- Some sounds are harder to synthesize
- Try simplifying the prompt

### Sounds Not Matching Prompt
- Be specific: "dog barking" instead of "dog"
- Use descriptive words: "high-pitched whistle" vs "whistle"
- Try variations: "bee buzzing" vs "swarm of bees"

### Generation Too Slow
- Reduce generations: `--gens 50`
- Reduce population: `--pop 20`
- Reduce duration: `--dur 1.0`

### Generation Too Fast, Low Quality
- Increase generations: `--gens 200`
- Increase population: `--pop 40`

## Tips for Best Results

1. **Be Specific**: "high-pitched bird chirp" > "bird"
2. **Use Audio Terms**: Mention pitch, rhythm, texture
3. **Start Simple**: Test with simple sounds first
4. **Iterate**: Try variations of your prompt
5. **Check Spectrogram**: Visual feedback on what was generated
6. **Adjust Duration**: Some sounds need more time to evolve
7. **Experiment**: The system learns what works through evolution

## Next Steps

- Check `IMPROVEMENTS.md` for technical details on accuracy
- Read `README.md` for full documentation
- Experiment with different prompts and settings
- Share your best results!

## Common Use Cases

### Sound Design
```bash
python main.py --prompt "sci-fi door open" --out sfx_door.wav
python main.py --prompt "laser gun shot" --out sfx_laser.wav
```

### Music Production
```bash
python main.py --prompt "deep bass drone" --dur 4.0 --out bass.wav
python main.py --prompt "pad sound evolving" --dur 8.0 --out pad.wav
```

### Foley/Sound Effects
```bash
python main.py --prompt "footsteps on gravel" --out foley_steps.wav
python main.py --prompt "paper rustling" --out foley_paper.wav
```

### Education/Research
```bash
python main.py --prompt "pure sine wave 440Hz" --out test_sine.wav
python main.py --prompt "white noise" --out test_noise.wav
```

Enjoy creating sounds!
