"""Search skeleton: random walk over voice-tensor space. Runs as-is, but
the fitness function and search strategy are deliberately naive — that is
your hour.

    python search.py --reference_dir ../reference --start blend_baseline.pt \
        --iters 150 --out voice.pt

Ideas the skeleton does NOT do for you:
  * fitness beyond raw similarity (naturalness terms, self-similarity
    across sentences, spectral sanity checks) — see the warning in
    similarity.py
  * evaluating on 2-3 DIFFERENT sentences per candidate (one sentence
    overfits)
  * annealing the step size; accepting sideways moves; restarts
  * structured perturbations: are all 256 dimensions doing the same kind
    of work? Perturb halves separately and find out.
  * the tensor is 510 rows of 256 dims — synthesizing a given text uses ONE
    row, picked by the text's phoneme count. Which rows does your fitness
    actually test, and what is randn_like doing to all the others? (This is
    why a local gain can evaporate on sentences you never evaluated.)
  * listening checkpoints: dump audio every N accepted steps and USE YOUR EARS
"""
import argparse

import torch
import cma
import numpy as np

import synth
import similarity as sim
import glob
import librosa 
# Load reference target clips provided in starter kit
ref_clips = glob.glob("reference/*.wav") #
ref_stats = []

for clip_path in ref_clips:
    ref_wav, sr = librosa.load(clip_path, sr=24000)
    ref_stats.append(prosody_stats(ref_wav, sr=24000))

# Average ground-truth prosody targets across all reference clips
target_prosody = {
    'f0_std': float(np.mean([s['f0_std'] for s in ref_stats])),
    'flatness': float(np.mean([s['flatness'] for s in ref_stats])),
    'rms_std': float(np.mean([s['rms_std'] for s in ref_stats]))
}

print(f"Target Prosody Targets -> F0 Std: {target_prosody['f0_std']:.2f}, "
      f"Flatness: {target_prosody['flatness']:.4f}, RMS Std: {target_prosody['rms_std']:.2f}")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty."
]


from scipy.signal import butter, filtfilt, find_peaks

def speaking_rate(wav, sr=24000, min_peak_distance_s=0.10, silence_floor_db=-40):
    """Estimate syllables/sec from raw audio using energy envelope peaks."""
    hop = 256
    rms = librosa.feature.rms(y=wav, hop_length=hop)[0]
    env_db = librosa.amplitude_to_db(rms, ref=np.max)

    frame_rate = sr / hop
    # Syllables land around 3-8 Hz; lowpass filter at 10 Hz
    b, a = butter(2, 10.0 / (frame_rate / 2), btype='low')
    smooth = filtfilt(b, a, env_db)

    min_distance = int(min_peak_distance_s * frame_rate)
    peaks, _ = find_peaks(smooth, height=silence_floor_db, prominence=3.0, distance=min_distance)

    voiced_duration = np.sum(smooth > silence_floor_db) / frame_rate
    return len(peaks) / voiced_duration if voiced_duration > 0 else 0.0


def prosody_stats(wav, sr=24000):
    """
    Extracts pitch variance, spectral flatness (raspiness), and dynamic range.
    Optimized for fast CPU evaluation in search loops.
    """
    # Speed Optimization: Downsample to 12kHz purely for fast F0 tracking
    wav_12k = librosa.resample(wav, orig_sr=sr, target_sr=12000)
    
    # Fast YIN pitch estimation
    f0 = librosa.yin(wav_12k, fmin=65, fmax=400, sr=12000, frame_length=1024, hop_length=256)
    
    # Strict voiced filtering: remove NaNs and boundary artifacts (unvoiced frames)
    voiced_mask = (~np.isnan(f0)) & (f0 > 67) & (f0 < 395)
    f0_voiced = f0[voiced_mask]
    
    f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 5 else 0.0

    # Spectral flatness (1.0 = noise/rasp, 0.0 = pure tone)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=wav)))
    
    # Loudness expressiveness / dynamic range
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=wav)[0], ref=np.max)
    rms_std = float(np.std(rms_db))

    return {'f0_std': f0_std, 'flatness': flatness, 'rms_std': rms_std}


def naturalness_score(wav, target_prosody, sr=24000):
    s = prosody_stats(wav, sr)
    def rel(a, b): return 1.0 - min(abs(a - b) / max(b, 1e-3), 1.0)
    
    return float(np.mean([
        rel(s['f0_std'], target_prosody['f0_std']),
        rel(s['flatness'], target_prosody['flatness']),
        rel(s['rms_std'], target_prosody['rms_std']),
    ]))


def fitness(voice_tensor, target_emb, texts, base_tensor_norm=None, target_prosody=None):
    generated_embs, target_sims, naturalness_scores, rate_ratios = [], [], [], []

    for t in texts:
        wav = synth.synthesize(t, voice_tensor)

        if wav is None or len(wav) == 0:
            return -1.0
        if np.max(np.abs(wav)) > 0.99:  # Audio clipping check
            return -0.5

        # Duration & Pace sanity checks
        expected_duration = len(t) / 15.0
        actual_duration = len(wav) / 24000.0
        rate_ratio = actual_duration / expected_duration
        rate_ratios.append(rate_ratio)
        
        if rate_ratio < 0.5 or rate_ratio > 2.0:
            return -0.2

        target_sims.append(sim.similarity_to_target(wav, target_emb))
        generated_embs.append(sim.embed(wav))
        
        if target_prosody is not None:
            naturalness_scores.append(naturalness_score(wav, target_prosody))

    avg_target_sim = float(np.mean(target_sims))
    avg_naturalness = float(np.mean(naturalness_scores)) if naturalness_scores else 0.5

    # Cross-utterance self-similarity
    self_sims = [sim.cosine_similarity(generated_embs[i], generated_embs[j])
                 for i in range(len(generated_embs)) for j in range(i + 1, len(generated_embs))]
    avg_self_sim = float(np.mean(self_sims)) if self_sims else 1.0

    # Pacing consistency across different sentences
    rate_consistency = 1.0 - min(float(np.std(rate_ratios)), 1.0) if len(rate_ratios) > 1 else 1.0

    # L2 Norm Regularization to prevent extreme tensor values
    norm_penalty = 0.0
    if base_tensor_norm is not None:
        if isinstance(voice_tensor, torch.Tensor):
            current_norm = float(torch.norm(voice_tensor).item())
        else:
            current_norm = float(np.linalg.norm(voice_tensor))
            
        norm_penalty = max(0.0, current_norm - (base_tensor_norm * 1.2)) * 0.1

    alpha, beta, gamma, delta = 0.70, 0.15, 0.10, 0.05

    return (alpha * avg_target_sim 
            + beta * avg_self_sim 
            + gamma * avg_naturalness 
            + delta * rate_consistency 
            - norm_penalty)


def _tensor_from_flat(x, shape, dtype):
    arr = np.asarray(x, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr).to(dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--start", required=True, help="starting .pt tensor")
    ap.add_argument("--iters", type=int, default=150,
          help="total candidate evaluations")
    ap.add_argument("--step", type=float, default=0.03,
          help="initial CMA sigma")
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--num_sentences", type=int, default=1,
            help="number of sentences from SENTENCES for fitness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=5)
    args = ap.parse_args()

    target = sim.target_embedding(args.reference_dir)
    texts = SENTENCES[:max(1, args.num_sentences)]

    best = synth.load_voice(args.start).clone()
    best_f = fitness(best, target, texts)
    print(f"start fitness: {best_f:.4f}")

    x0 = best.detach().cpu().numpy().astype(np.float64).ravel()
    es = cma.CMAEvolutionStrategy(
        x0,
        args.step,
        {
            "popsize": args.popsize,
            "seed": args.seed,
            "verbose": -9,
            # Full covariance is too expensive at this dimensionality.
            "CMA_diagonal": True,
        },
    )

    accepted = 0
    evals = 0
    generation = 0
    while evals < args.iters and not es.stop():
        generation += 1
        batch = min(args.popsize, args.iters - evals)
        xs = es.ask(batch)
        losses = []

        for x in xs:
            cand = _tensor_from_flat(x, best.shape, best.dtype)
            try:
                f = fitness(cand, target, texts)
            except Exception as err:
                # Keep search moving if one candidate hits a synthesis edge case.
                f = -1e9
                print(f"eval {evals + 1:4d} failed: {err}")

            losses.append(-f)  # CMA minimizes
            evals += 1

            if f > best_f:
                best = cand.clone()
                best_f = f
                accepted += 1
                print(
                    f"eval {evals:4d}  accepted #{accepted}  "
                    f"fitness {best_f:.4f}"
                )
                if accepted % args.listen_every == 0:
                    import soundfile as sf
                    sf.write(
                        f"listen_{accepted}.wav",
                        synth.synthesize(texts[0], best),
                        synth.SR,
                    )
                    print(f"  -> wrote listen_{accepted}.wav - GO LISTEN")

        es.tell(xs, losses)
        print(
            f"gen {generation:3d}  evals {evals:4d}/{args.iters}  "
            f"sigma {es.sigma:.5f}  best {best_f:.4f}"
        )

    torch.save(best, args.out)
    import soundfile as sf
    sf.write("listen_final.wav", synth.synthesize(SENTENCES[0], best), synth.SR)
    print(f"final fitness {best_f:.4f} -> saved {args.out}")
    print("wrote listen_final.wav — LISTEN BEFORE YOU SUBMIT")


if __name__ == "__main__":
    main()
