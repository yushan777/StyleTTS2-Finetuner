# Scans a folder of .wav files and prints a duration report useful for choosing
# StyleTTS2's max_len config value. Shows total/average duration, a histogram of
# clip lengths bucketed in 2-second bands (with the equivalent mel-frame counts),
# and the 10 longest clips. Run: python audio_duration_report.py <wav_folder>

import os
import sys
import soundfile as sf
from collections import defaultdict

# Must match the trainer config — used to convert seconds ↔ mel spectrogram frames
HOP_LENGTH = 300
SAMPLE_RATE = 24000

def secs_to_frames(s):
    return int(s * SAMPLE_RATE / HOP_LENGTH)

def report(folder, bucket_size=2):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    if not files:
        print(f"No wav files found in {folder}")
        return

    durations = []
    for f in sorted(files):
        try:
            info = sf.info(os.path.join(folder, f))
            durations.append((f, info.duration))
        except Exception as e:
            print(f"  [error] {f}: {e}")

    # Group clips into fixed-width duration buckets (default 2 s wide)
    buckets = defaultdict(list)
    for f, d in durations:
        bucket = int(d // bucket_size) * bucket_size
        buckets[bucket].append((f, d))

    total = sum(d for _, d in durations)
    max_bucket = max(buckets)

    print(f"\nFolder : {folder}")
    print(f"Files  : {len(durations)}")
    print(f"Total  : {total/60:.1f} mins ({total:.0f}s)")
    print(f"Average: {total/len(durations):.1f}s  ({secs_to_frames(total/len(durations))} frames)")
    print()
    print(f"Note: 'Frames' are mel spectrogram frames — the unit StyleTTS2 uses internally to")
    print(f"      measure audio length. Each frame = {HOP_LENGTH}/{SAMPLE_RATE}s = {1000*HOP_LENGTH//SAMPLE_RATE}ms of audio (hop_length={HOP_LENGTH}, sr={SAMPLE_RATE}).")
    print(f"      These values are specific to this trainer's config and will differ for other trainers.")
    print(f"      The config's max_len setting is in frames — use the table below to find the equivalent seconds.")
    print()
    print(f"{'Range':<22} {'Frames':<16} {'Count':>6}  {'% of total':>10}")
    print("-" * 58)

    for b in range(0, max_bucket + bucket_size, bucket_size):
        items = buckets.get(b, [])
        label = f"{b}s – {b+bucket_size}s"
        frames = f"{secs_to_frames(b)} – {secs_to_frames(b+bucket_size)}"
        pct = 100 * len(items) / len(durations)
        print(f"  {label:<20} {frames:<16} {len(items):>6}   {pct:>6.1f}%")

    print()
    longest = sorted(durations, key=lambda x: x[1], reverse=True)[:10]
    print("Longest clips:")
    for f, d in longest:
        print(f"  {d:6.2f}s  ({secs_to_frames(d):>5} frames)  {f}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    report(folder)
