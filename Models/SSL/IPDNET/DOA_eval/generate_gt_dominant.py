import os
import re
import pandas as pd


def generate_ground_truth_csv_one_doa(folder_path, output_csv_path):
    pattern = re.compile(r"mic_fileid_(\d+)_doa(\d+)_.*\.wav")

    rows = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)

        if match:
            fileid = int(match.group(1))
            doa = int(match.group(2))

            rows.append({
                "index": fileid,
                "doa": doa,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("index").reset_index(drop=True)

    df.to_csv(output_csv_path, index=False)

    print(f"Saved to: {output_csv_path}")
    return df


folder_path = r"D:\邵鹏远\UCL\博1\code\Whisper_ASR\data\dataset_4mic_3spk_dominant\Eval\mic"
output_csv_path = r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet2\inference_results_dominant\ground_truth.csv"

df = generate_ground_truth_csv_one_doa(folder_path, output_csv_path)

print(df)