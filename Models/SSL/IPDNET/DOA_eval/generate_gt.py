import os
import re
import pandas as pd


def generate_ground_truth_excel(folder_path, output_excel_path):
    pattern = re.compile(r"mic_fileid_(\d+)_doa(\d+)_3spk\.wav")

    data = {}

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)

        if match:
            fileid = int(match.group(1))
            doa = int(match.group(2))

            if fileid not in data:
                data[fileid] = []

            data[fileid].append(doa)

    rows = []

    for fileid in sorted(data.keys()):
        doas = sorted(data[fileid])

        if len(doas) != 3:
            print(f"Warning: fileid {fileid} has {len(doas)} DOAs: {doas}")

        row = {
            "index": fileid,
            "doa1": doas[0] if len(doas) > 0 else None,
            "doa2": doas[1] if len(doas) > 1 else None,
            "doa3": doas[2] if len(doas) > 2 else None,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_excel_path, index=False)

    print(f"Saved to: {output_excel_path}")
    return df


folder_path = r"D:\邵鹏远\UCL\博1\code\Whisper_ASR\data\dataset_4mic_3spk\Eval\mic"
output_excel_path = r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet2\inference_results\ground_truth.csv"

df = generate_ground_truth_excel(folder_path, output_excel_path)

print(df)