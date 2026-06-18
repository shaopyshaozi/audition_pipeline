import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import permutations
from tqdm import tqdm

def convert_pred_angle(pred):
    return np.asarray(pred) % 360


def circular_angle_diff(a, b):
    """
    Smallest angular difference between two angles in degrees.
    Output range: [0, 180]
    """
    return np.abs((a - b + 180) % 360 - 180)


def circular_mean_deg_360(angles_deg, weights=None):
    angles_deg = convert_pred_angle(np.asarray(angles_deg))
    angles_rad = np.deg2rad(angles_deg)

    if weights is None:
        weights = np.ones_like(angles_rad)

    x = np.sum(weights * np.cos(angles_rad))
    y = np.sum(weights * np.sin(angles_rad))

    mean_angle = np.rad2deg(np.arctan2(y, x))
    return mean_angle % 360


def best_match_doa(pred_doas, gt_doas, threshold=5):
    """
    Match predicted DOAs to GT DOAs using the best permutation.
    Handles different order between prediction and GT.
    """

    pred_doas = np.asarray(pred_doas, dtype=float)
    gt_doas = np.asarray(gt_doas, dtype=float)

    n_pred = len(pred_doas)
    n_gt = len(gt_doas)

    if n_pred == 0:
        return [], 0, n_gt, []

    best_pairs = None
    best_total_error = np.inf

    for pred_indices in permutations(range(n_pred), min(n_pred, n_gt)):
        used_gt_indices = range(min(n_pred, n_gt))

        total_error = 0
        pairs = []

        for pi, gi in zip(pred_indices, used_gt_indices):
            err = circular_angle_diff(pred_doas[pi], gt_doas[gi])
            total_error += err
            pairs.append((pi, gi, err))

        if total_error < best_total_error:
            best_total_error = total_error
            best_pairs = pairs

    correct_pairs = [
        pair for pair in best_pairs
        if pair[2] <= threshold
    ]

    num_correct = len(correct_pairs)
    num_missed = n_gt - num_correct
    correct_errors = [pair[2] for pair in correct_pairs]

    return best_pairs, num_correct, num_missed, correct_errors


def postprocess_one_sample(
    doaest_path,
    vadest_path,
    sample_idx=0,
    num_sources=3,
    vad_th=0.5,
    score_mode="similarity",
    min_points_per_source=3,
    plot=True,
    plot_cluster=True,
):
    doa_est = np.load(doaest_path)   # [B, T, 2, K]
    vad_est = np.load(vadest_path)   # [B, T, K]

    azi = convert_pred_angle(doa_est[sample_idx, :, 1, :])  # [T, K]
    score = vad_est[sample_idx, :, :]                       # [T, K]

    if score_mode == "similarity":
        active = score > vad_th
    elif score_mode == "distance":
        active = score < vad_th
    else:
        raise ValueError("score_mode must be 'similarity' or 'distance'")

    T, K = azi.shape
    time_axis = np.arange(T) * 0.1

    # -----------------------------
    # 1. Plot original predicted tracks
    # -----------------------------
    if plot:
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
    valid_times = []

    for t in range(T):
        for k in range(K):
            if active[t, k]:
                valid_angles.append(azi[t, k])
                if score_mode == "similarity":
                    valid_weights.append(max(float(score[t, k]), 1e-6))
                else:
                    valid_weights.append(1.0 / (float(score[t, k]) + 1e-6))
                valid_times.append(t)

    valid_angles = np.array(valid_angles)
    valid_weights = np.array(valid_weights)
    valid_times = np.array(valid_times)

    if len(valid_angles) < num_sources:
        print("Not enough reliable DOA points.")
        return []

    # -----------------------------
    # 3. Circular clustering
    # -----------------------------
    angle_rad = np.deg2rad(valid_angles)

    X = np.stack(
        [
            np.cos(angle_rad),
            np.sin(angle_rad),
        ],
        axis=1,
    )

    kmeans = KMeans(n_clusters=num_sources, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X, sample_weight=valid_weights)

    # -----------------------------
    # 4. Plot clustered points
    # -----------------------------
    if plot_cluster:
        plt.figure(figsize=(10, 4))

        for src_id in range(num_sources):
            cluster_times = valid_times[labels == src_id] * 0.1
            cluster_angles = valid_angles[labels == src_id]

            plt.plot(
                cluster_times,
                cluster_angles,
                marker="o",
                linestyle="",
                label=f"Cluster {src_id}",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Estimated azimuth (degree, 0-360)")
        plt.title("Clustered predicted DOA points")
        plt.ylim(0, 360)
        plt.yticks(np.arange(0, 361, 45))
        plt.legend()
        plt.grid(True)
        plt.show()

    # -----------------------------
    # 5. Compute one mean DOA per cluster
    # -----------------------------
    final_doas = []

    for src_id in range(num_sources):
        cluster_angles = valid_angles[labels == src_id]
        cluster_weights = valid_weights[labels == src_id]

        if len(cluster_angles) < min_points_per_source:
            continue

        doa = circular_mean_deg_360(cluster_angles, cluster_weights)
        final_doas.append(doa)

    final_doas = sorted(final_doas)

    # print("Final one DOA per source:")
    # for i, doa in enumerate(final_doas):
    #     print(f"Source {i}: {doa:.2f} degrees")

    return final_doas


def evaluate_one_file_with_csv(
    doaest_path,
    vadest_path,
    csv_path,
    scene_index,
    sample_idx=0,
    num_sources=3,
    vad_th=0.5,
    score_mode="similarity",
    threshold=5,
    plot=True,
    plot_cluster=True,
):
    df = pd.read_csv(csv_path)

    row = df[df["index"] == scene_index].iloc[0]

    # Now only one GT DOA
    gt_doa = float(row["doa"])
    gt_doa = convert_pred_angle(gt_doa)

    pred_doas = postprocess_one_sample(
        doaest_path=doaest_path,
        vadest_path=vadest_path,
        sample_idx=sample_idx,
        num_sources=num_sources,
        vad_th=vad_th,
        score_mode=score_mode,
        plot=plot,
        plot_cluster=plot_cluster,
    )

    errors = [float(circular_angle_diff(pred_doa, gt_doa)) for pred_doa in pred_doas]

    best_idx = int(np.argmin(errors))
    best_pred_doa = pred_doas[best_idx]
    best_error = errors[best_idx]

    correct = best_error <= threshold

    # print("\nGround truth DOA:")
    # print(f"{gt_doa:.2f}°")

    # print("\nPredicted DOAs:")
    # print(np.round(pred_doas, 2))

    # print("\nBest matching:")
    # print(
    #     f"Pred {best_pred_doa:.2f}°  <->  "
    #     f"GT {gt_doa:.2f}°  "
    #     f"error = {best_error:.2f}°"
    # )

    return {
        "gt_doa": gt_doa,
        "pred_doas": pred_doas,
        "best_pred_doa": best_pred_doa,
        "error": [best_error],
        "correct": correct,
    }


results=[]
threshold = 10
for scene_index in tqdm(range(984)):
    result = evaluate_one_file_with_csv(
        doaest_path=fr"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet\inference_results_dominant\{scene_index}_doaest.npy",
        vadest_path=fr"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet\inference_results_dominant\{scene_index}_vadest.npy",
        csv_path=r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet\inference_results_dominant\ground_truth.csv",
        scene_index=scene_index,
        sample_idx=0,
        num_sources=3,
        vad_th=0.7,
        score_mode="similarity",
        threshold=threshold,
        plot=False,
        plot_cluster=False,
    )

    results+=result['error']

results = np.array(results)

mean_error_all = np.mean(results)

correct_errors = results[results <= threshold]

acc_5deg = len(correct_errors) / len(results)
mae_5deg = np.mean(correct_errors) if len(correct_errors) > 0 else np.nan

print(f"Mean angle error across all examples: {mean_error_all:.4f}")
print(f"Accuracy within {threshold} degrees: {acc_5deg:.4f}")
print(f"MAE of errors within {threshold} degrees: {mae_5deg:.4f}")
