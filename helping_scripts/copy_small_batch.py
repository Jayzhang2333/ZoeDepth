#!/usr/bin/env python3
import os
import shutil
import argparse

def copy_and_record(input_txt: str, dest_dir: str, output_txt: str):
    """
    Read each line of `input_txt`, expecting exactly three file paths per line.
    Copy those files into `dest_dir`, and write a corresponding line to `output_txt`
    listing the three new locations (space‐separated). Lines with missing files
    or not exactly three tokens are skipped (with a warning).
    """
    os.makedirs(dest_dir, exist_ok=True)

    with open(input_txt, 'r') as fin, open(output_txt, 'w') as fout:
        for lineno, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 3:
                print(f"[Warning] Line {lineno} does not have exactly 3 paths → skipped.")
                continue

            # Verify all three sources exist
            missing = [p for p in parts if not os.path.isfile(p)]
            if missing:
                print(f"[Warning] Line {lineno}, missing files: {missing} → skipped.")
                continue

            # Build destination paths and copy
            dest_paths = []
            for src in parts:
                fname = os.path.basename(src)
                dst = os.path.join(dest_dir, fname)
                try:
                    shutil.copy2(src, dst)
                    dest_paths.append(dst)
                except Exception as e:
                    print(f"[Error] Copying '{src}' failed: {e}")
                    dest_paths = []
                    break

            # If all three copied successfully, record them
            if len(dest_paths) == 3:
                fout.write(" ".join(dest_paths) + "\n")
                print(f"Line {lineno} → copied and recorded: {dest_paths}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy files (three per line) from an input .txt to a destination folder, "
                    "and write a new .txt listing their new locations."
    )
    parser.add_argument(
        "input_txt",
        help="Path to the input text file. Each non‐empty line must have exactly three file paths."
    )
    parser.add_argument(
        "dest_dir",
        help="Directory into which all files will be copied."
    )
    parser.add_argument(
        "output_txt",
        help="Path to the output text file that will list the three new locations per line."
    )
    args = parser.parse_args()

    copy_and_record(args.input_txt, args.dest_dir, args.output_txt)
