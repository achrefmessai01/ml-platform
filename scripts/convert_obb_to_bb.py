import os
import numpy as np

def obb_to_aabb_line(line: str) -> str:
    parts = line.strip().split()
    cls = parts[0]
    coords = np.array(parts[1:], dtype=float).reshape(4, 2)
    xmin, ymin = coords[:,0].min(), coords[:,1].min()
    xmax, ymax = coords[:,0].max(), coords[:,1].max()
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def convert_obb_folder_inplace(folder: str):
    txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    for fname in txt_files:
        path = os.path.join(folder, fname)
        with open(path, 'r') as fin:
            lines = [l.strip() for l in fin if l.strip()]
        converted = [obb_to_aabb_line(l) for l in lines]
        with open(path, 'w') as fout:
            fout.write('\n'.join(converted) + '\n')
    print(f"Converted {len(txt_files)} files in '{folder}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert YOLO‑OBB labels to standard YOLO in-place")
    parser.add_argument('--folder', '-f', required=True,
                        help="Path to folder containing YOLO‑OBB .txt labels")
    args = parser.parse_args()
    convert_obb_folder_inplace(args.folder)
