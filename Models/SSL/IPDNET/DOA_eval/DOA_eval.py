import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def convert_pred_angle(pred):
    """
    Convert angle from [-180, 180] to [0, 360).

    Examples:
    -10  -> 350
    -90  -> 270
    -180 -> 180
    0    -> 0
    90   -> 90
    """
    return pred % 360


def circular_mean_deg_360(angles_deg, weights=None):
    angles_deg = convert_pred_angle(np.asarray(angles_deg))
    angles_rad = np.deg2rad(angles_deg)

    if weights is None:
        weights = np.ones_like(angles_rad)

    x = np.sum(weights * np.cos(angles_rad))
    y = np.sum(weights * np.sin(angles_rad))

    mean_angle = np.rad2deg(np.arctan2(y, x))
    return convert_pred_angle(mean_angle)


def postprocess_one_sample(
    doaest_path,
    vadest_path,
    sample_idx=0,
    num_sources=2,
    vad_th=0.4,
    min_points_per_source=3,
):
    doa_est = np.load(doaest_path)   # [B, T, 2, K]
    vad_est = np.load(vadest_path)   # [B, T, K]

    # Azimuth only, convert from [-180, 180] to [0, 360)
    azi = convert_pred_angle(doa_est[sample_idx, :, 1, :])  # [T, K]
    score = vad_est[sample_idx, :, :]                       # [T, K]

    # Your vad_est is MSE-like: smaller means more confident
    active = score < vad_th

    T, K = azi.shape
    time_axis = np.arange(T) * 0.1  # 100 ms per frame

    # -----------------------------
    # 1. Visualize raw trajectories
    # -----------------------------
    plt.figure(figsize=(10, 4))

    for k in range(K):
        y = np.where(active[:, k], azi[:, k], np.nan)
        plt.plot(time_axis, y, marker="o", label=f"Pred track {k}")

    plt.xlabel("Time (s)")
    plt.ylabel("Estimated azimuth (degree, 0-360)")
    plt.title("Frame-wise predicted DOA trajectory")
    plt.ylim(0, 360)
    plt.yticks(np.arange(0, 361, 45))
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------------
    # 2. Collect reliable DOA points
    # -----------------------------
    valid_angles = []
    valid_weights = []

    for t in range(T):
        for k in range(K):
            if active[t, k]:
                angle = azi[t, k]
                valid_angles.append(angle)

                # smaller MSE = larger confidence
                weight = 1.0 / (score[t, k] + 1e-6)
                valid_weights.append(weight)

    valid_angles = np.array(valid_angles)
    valid_weights = np.array(valid_weights)

    if len(valid_angles) < num_sources:
        print("Not enough reliable DOA points.")
        return []

    # -----------------------------
    # 3. Circular clustering
    # -----------------------------
    angle_rad = np.deg2rad(valid_angles)
    X = np.stack([np.cos(angle_rad), np.sin(angle_rad)], axis=1)

    kmeans = KMeans(n_clusters=num_sources, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X, sample_weight=valid_weights)

    final_doas = []

    for src_id in range(num_sources):
        cluster_angles = valid_angles[labels == src_id]
        cluster_weights = valid_weights[labels == src_id]

        if len(cluster_angles) < min_points_per_source:
            continue

        doa = circular_mean_deg_360(cluster_angles, cluster_weights)
        final_doas.append(doa)

    final_doas = sorted(final_doas)

    print("Final one DOA per source:")
    for i, doa in enumerate(final_doas):
        print(f"Source {i}: {doa:.2f} degrees")

    return final_doas


final_doas = postprocess_one_sample(
    doaest_path=r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet2\inference_results_dominant\4_doaest.npy",
    vadest_path=r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet2\inference_results_dominant\4_vadest.npy",
    sample_idx=0,
    num_sources=3,
    vad_th=0.2,
)