import pandas as pd

# =========================
# Config
# =========================
CSV_PATH = (
    "/mnt/d/邵鹏远/UCL/博1/code/"
    "audition_pipeline/offline/results/"
    "pipeline_whisper_small_wer_details_1asr.csv"
)

DOA_THRESHOLD_DEG = 20.0  # <- change this----------------------------------------------------------

# =========================
# Load CSV
# =========================
df = pd.read_csv(CSV_PATH)

# Ensure numeric
df["selected_doa_error_deg"] = pd.to_numeric(
    df["selected_doa_error_deg"], errors="coerce"
)
df["wer"] = pd.to_numeric(df["wer"], errors="coerce")

# Remove rows with missing values
valid_df = df.dropna(subset=["selected_doa_error_deg", "wer"])

# =========================
# Filter by DOA error
# =========================
subset = valid_df[
    valid_df["selected_doa_error_deg"] <= DOA_THRESHOLD_DEG
]

# =========================
# Statistics
# =========================
total_samples = len(valid_df)
selected_samples = len(subset)

percentage = (
    selected_samples / total_samples * 100
    if total_samples > 0 else 0
)

mean_wer = subset["wer"].mean()

# Optional weighted WER if available
weighted_wer = None
if {"edit_distance", "ref_words"}.issubset(valid_df.columns):
    total_edits = subset["edit_distance"].sum()
    total_words = subset["ref_words"].sum()

    if total_words > 0:
        weighted_wer = total_edits / total_words

# =========================
# Output
# =========================
print(f"DOA threshold: {DOA_THRESHOLD_DEG:.1f}°")
print(f"Total samples: {total_samples}")
print(f"Samples within threshold: {selected_samples}")
print(f"Percentage within threshold: {percentage:.2f}%")
print()

print(f"Mean sample WER: {mean_wer:.4f}")

if weighted_wer is not None:
    print(f"Corpus WER: {weighted_wer:.4f}")