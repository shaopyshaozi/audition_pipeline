import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

csv_path = "/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/results/pipeline_realtime_small_details_1asr.csv"

df = pd.read_csv(csv_path)

errors = pd.to_numeric(
    df["selected_doa_error_deg"],
    errors="coerce"
).dropna()

# Keep only 0–180 degrees
errors = errors[(errors >= 0) & (errors <= 180)]

x = np.linspace(0, 180, 1000)
kde = gaussian_kde(errors)

plt.figure(figsize=(8, 5))

# Histogram
plt.hist(
    errors,
    bins=np.arange(0, 181, 5),
    density=True,
    alpha=0.4,
    edgecolor="black",
    label="Histogram"
)

# Smooth KDE curve
plt.plot(
    x,
    kde(x),
    linewidth=2,
    label="KDE"
)

plt.xlim(0, 180)
plt.xlabel("Selected DOA Error (degrees)")
plt.ylabel("Density")
plt.title("Selected DOA Error Distribution (0–180°)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("selected_doa_error_hist_kde_0_180.png", dpi=300)
plt.show()