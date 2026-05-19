import sys
import os
import glob
import phonemizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab', quiet=True)

backend = phonemizer.backend.EspeakBackend(
    language='en-us', preserve_punctuation=True, with_stress=True)

def is_already_phonemized(lines):
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            return not parts[1].isascii()
    return False

def convert(path):
    with open(path) as f:
        lines = f.readlines()

    if is_already_phonemized(lines):
        print(f"  [skip] {path} appears already phonemized (IPA characters detected)")
        return

    out = []
    for i, line in enumerate(lines):
        parts = line.strip().split('|')
        if len(parts) < 3:
            print(f"  [skip] line {i+1}: unexpected format: {line.strip()}")
            out.append(line)
            continue

        filename, text, speaker_id = parts[0], parts[1], parts[2]
        ps = backend.phonemize([text])
        ps = word_tokenize(ps[0])
        phonemes = ' '.join(ps)
        out.append(f"{filename}|{phonemes}|{speaker_id}\n")

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(lines)} done...")

    stem, ext = os.path.splitext(path)
    out_path = f"{stem}_phonemized{ext}"
    with open(out_path, 'w') as f:
        f.writelines(out)

    print(f"  Done: {out_path} ({len(out)} lines)")

def find_list_files(directory):
    matches = []
    for pattern in ("*train_list.txt", "*val_list.txt"):
        matches.extend(sorted(glob.glob(os.path.join(directory, pattern))))
    return matches

if __name__ == "__main__":
    if len(sys.argv) > 1:
        lists = []
        for arg in sys.argv[1:]:
            if os.path.isdir(arg):
                found = find_list_files(arg)
                if not found:
                    print(f"No list files found in: {arg}")
                lists.extend(found)
            else:
                lists.append(arg)
    else:
        base = "training_datasets"
        lists = []
        if os.path.isdir(base):
            for dataset in sorted(os.listdir(base)):
                dataset_dir = os.path.join(base, dataset)
                if os.path.isdir(dataset_dir):
                    lists.extend(find_list_files(dataset_dir))
        if not lists:
            print(f"No list files found under {base}/")
            sys.exit(0)

    for path in lists:
        if not os.path.exists(path):
            print(f"Not found: {path}")
            continue
        print(f"Converting: {path}")
        convert(path)
