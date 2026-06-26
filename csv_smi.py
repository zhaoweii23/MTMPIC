import csv
import os
import re

def csv_to_single_smi(input_csv, output_dir, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    records = []

    with open(input_csv, 'r', encoding='utf-8-sig') as f:

        first_line = f.readline().strip()
        f.seek(0)
        print(f"CSV first line: {first_line}")


        if first_line == 'SMILES,DICL ID':
            reader = csv.reader(f)
            next(reader)  # skip header row
            print("Header row skipped.")
        else:
            reader = csv.reader(f)

        for row in reader:
            if not row or len(row) < 2:
                continue
            smiles = row[19].strip()
            title = row[0].strip()

            smiles = smiles.replace('^M', '').replace('\r', '').replace('\n', '')
            title = title.replace('^M', '').replace('\r', '').replace('\n', '')

            if not smiles:
                print(f"Warning: row {reader.line_num} SMILES is empty, skipping.")
                continue

            if not title:
                title = f"mol_{len(records)+1}"


            title = re.sub(r'[\\/*?:"<>|]', '_', title)
            records.append((smiles, title))

    if not records:
        print("No valid data found. Please check the CSV format.")
        return

    print(f"Read {len(records)} molecules, start generating .smi files...")

    for idx, (smiles, title) in enumerate(records, 1):
        filename = f"{title}.smi"
        filepath = os.path.join(output_dir, filename)


        if os.path.exists(filepath) and not overwrite:
            base, ext = os.path.splitext(filename)
            new_name = f"{base}_{idx}{ext}"
            filepath = os.path.join(output_dir, new_name)
            print(f"File {filename} already exists, using {new_name} instead.")


        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"{smiles} {title}\n")
        print(f"Written: {filepath} -> {smiles[:50]}...")

    print(f"Successfully generated {len(records)} .smi files in {output_dir}")

if __name__ == "__main__":
    csv_file = "GABAA9CRS.csv"
    out_folder = "ligprep_input"
    csv_to_single_smi(csv_file, out_folder, overwrite=True)
