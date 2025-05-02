import os
import random
import pandas as pd
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1

def extract_sequence_from_pdb(filepath):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", filepath)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    try:
                        sequence += seq1(residue.resname)
                    except:
                        continue
            break
        break
    return sequence

def process_folder(data_dir='data/raw_pdbs'):
    class_map = {'alz': 'alzheimer', 'parkinson': 'parkinson', 'normal': 'normal'}
    file_groups = {'alzheimer': [], 'parkinson': [], 'normal': []}

    for fname in os.listdir(data_dir):
        for key in class_map:
            if key in fname.lower():
                file_groups[class_map[key]].append(os.path.join(data_dir, fname))
                break

    def split_group(files):
        random.shuffle(files)
        n = len(files)
        return {
            'train': files[:int(0.7 * n)],
            'val': files[int(0.7 * n):int(0.85 * n)],
            'test': files[int(0.85 * n):]
        }

    splits = {'train': [], 'val': [], 'test': []}

    for label, files in file_groups.items():
        split = split_group(files)
        for key in splits:
            for f in split[key]:
                seq = extract_sequence_from_pdb(f)
                if len(seq) >= 10:
                    splits[key].append({'sequence': seq, 'label': label})

    for key in splits:
        df = pd.DataFrame(splits[key])
        df.to_csv(f'data/{key}.csv', index=False)
        print(f"Saved {len(df)} records to data/{key}.csv")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    process_folder()
