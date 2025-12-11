# optimize.py
import numpy as np
import copy
import soundfile as sf
from tqdm import trange

from synth import synth, SR
from embed import get_audio_embedding_from_array, get_text_embedding

def random_params():
    """Return randomized parameter dict within reasonable ranges."""
    return {
        'base_freq': float(np.random.uniform(80, 1200)),
        'osc2_freq_ratio': float(np.random.uniform(0.5, 3.0)),
        'osc_mix': float(np.random.uniform(0.0, 1.0)),
        'osc1_wave': np.random.choice(['sine','saw','square','triangle']),
        'osc2_wave': np.random.choice(['sine','saw','square','triangle']),
        'lfo_rate': float(np.random.uniform(0.1, 8.0)),
        'lfo_amt': float(np.random.uniform(0.0, 0.5)),
        'noise_amp': float(np.random.uniform(0.0, 0.3)),
        'env_attack': float(np.random.uniform(0.001, 0.3)),
        'env_decay': float(np.random.uniform(0.01, 0.6)),
        'env_sustain': float(np.random.uniform(0.1, 0.9)),
        'env_release': float(np.random.uniform(0.01, 0.8)),
        'lp_cutoff': float(np.random.uniform(500.0, 18000.0)),
        'hp_cutoff': float(np.random.uniform(20.0, 300.0)),
        'filter_res': float(np.random.uniform(0.3, 3.0)),
        'fm_amount': float(np.random.uniform(0.0, 5.0)),
        'ring_mod': float(np.random.uniform(0.0, 0.5)),
        'harmonic_spread': float(np.random.uniform(0.0, 0.4)),
        'gain': float(np.random.uniform(0.4, 0.95)),
    }

def perturb_param(p, scale=0.1):
    q = p.copy()
    # numeric params: add gaussian noise (proportional to range)
    for k in ['base_freq','osc2_freq_ratio','osc_mix','lfo_rate','lfo_amt','noise_amp',
              'env_attack','env_decay','env_sustain','env_release','lp_cutoff','hp_cutoff',
              'filter_res','fm_amount','ring_mod','harmonic_spread','gain']:
        if k in q:
            val = q[k]
            # adapt noise scale by value magnitude
            if val == 0:
                q[k] = abs(val + np.random.randn()*scale)
            else:
                q[k] = float(val * (1 + np.random.randn()*scale))
    # categorical: small chance to flip waveform
    if np.random.rand() < 0.15:
        q['osc1_wave'] = np.random.choice(['sine','saw','square','triangle'])
    if np.random.rand() < 0.15:
        q['osc2_wave'] = np.random.choice(['sine','saw','square','triangle'])
    # clamp ranges
    q['base_freq'] = float(np.clip(q['base_freq'], 20, 8000))
    q['osc2_freq_ratio'] = float(np.clip(q['osc2_freq_ratio'], 0.1, 6.0))
    q['osc_mix'] = float(np.clip(q['osc_mix'], 0.0, 1.0))
    q['lfo_rate'] = float(np.clip(q['lfo_rate'], 0.01, 20.0))
    q['lfo_amt'] = float(np.clip(q['lfo_amt'], 0.0, 1.0))
    q['noise_amp'] = float(np.clip(q['noise_amp'], 0.0, 1.0))
    q['env_attack'] = float(np.clip(q['env_attack'], 0.0005, 1.0))
    q['env_decay'] = float(np.clip(q['env_decay'], 0.001, 2.0))
    q['env_sustain'] = float(np.clip(q['env_sustain'], 0.0, 1.0))
    q['env_release'] = float(np.clip(q['env_release'], 0.001, 3.0))
    q['lp_cutoff'] = float(np.clip(q['lp_cutoff'], 100.0, SR/2-100.0))
    q['hp_cutoff'] = float(np.clip(q['hp_cutoff'], 20.0, 1000.0))
    q['filter_res'] = float(np.clip(q['filter_res'], 0.1, 5.0))
    q['fm_amount'] = float(np.clip(q['fm_amount'], 0.0, 10.0))
    q['ring_mod'] = float(np.clip(q['ring_mod'], 0.0, 1.0))
    q['harmonic_spread'] = float(np.clip(q['harmonic_spread'], 0.0, 1.0))
    q['gain'] = float(np.clip(q['gain'], 0.01, 1.0))
    return q

def optimize_for_prompt(prompt, gens=120, pop_size=30, dur=2.0, verbose=True):
    text_emb = get_text_embedding(prompt)
    # if text_emb is a 1D vector shape unknown; unify to 1D numpy
    if hasattr(text_emb, 'cpu'):
        text_emb = text_emb.cpu().numpy().reshape(-1)
    else:
        text_emb = np.asarray(text_emb).reshape(-1)

    # initialize population
    population = [random_params() for _ in range(pop_size)]
    best = None
    best_score = -1e9
    best_audio = None
    history = []
    no_improvement_count = 0

    for g in trange(gens, desc="Generations"):
        candidates = []
        scores = []
        
        # Adaptive mutation scale based on progress
        if no_improvement_count > 15:
            mutation_scale = 0.25  # Larger mutations to escape local optima
        elif no_improvement_count > 8:
            mutation_scale = 0.18
        else:
            mutation_scale = 0.12
        
        for i in range(pop_size):
            # create variant
            if g == 0:
                cand = population[i]
            else:
                # Mix strategies: exploit best or explore population
                if np.random.rand() < 0.7:
                    cand = perturb_param(best, scale=mutation_scale)
                else:
                    parent = population[np.random.randint(pop_size)]
                    cand = perturb_param(parent, scale=mutation_scale * 1.3)
            
            # synth audio
            audio = synth(cand, dur=dur, sr=SR)
            # audio embedding
            emb = get_audio_embedding_from_array(audio, SR)
            # compute similarity (cosine)
            try:
                score = float(np.dot(emb, text_emb) / (np.linalg.norm(emb)*np.linalg.norm(text_emb) + 1e-9))
            except Exception:
                # fallback: negative distance
                score = -np.linalg.norm(emb - text_emb)
            candidates.append((cand, audio))
            scores.append(score)
        
        # select best in generation
        idx = int(np.argmax(scores))
        gen_best_score = scores[idx]
        gen_best_params, gen_best_audio = candidates[idx]
        history.append((gen_best_score, gen_best_params))
        
        if gen_best_score > best_score:
            best_score = gen_best_score
            best = gen_best_params
            best_audio = gen_best_audio
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Elite selection with diversity preservation
        sorted_idx = np.argsort(scores)[::-1]
        new_pop = []
        
        # Keep top performers
        elite_size = max(6, pop_size//4)
        for j in range(elite_size):
            new_pop.append(candidates[sorted_idx[j]][0])
        
        # Add some random diversity
        num_random = max(2, pop_size//10)
        for _ in range(num_random):
            new_pop.append(random_params())
        
        # Fill rest by perturbing elite
        while len(new_pop) < pop_size:
            parent_idx = np.random.randint(elite_size)
            parent = candidates[sorted_idx[parent_idx]][0]
            new_pop.append(perturb_param(parent, scale=mutation_scale))
        
        population = new_pop
        
        if verbose and (g % 10 == 0 or g == gens-1):
            print(f"[gen {g}] best={gen_best_score:.4f} global={best_score:.4f} no_improve={no_improvement_count}")
    
    return best, best_audio, history
