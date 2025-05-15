import numpy as np
import torch


def calc_angular_dist(pts_1: torch.Tensor, pts_2: torch.Tensor) -> torch.Tensor:
    """
    Computes angular distance (in radians) between two tensors of unit vectors.

    Args:
        pts_1: Tensor of shape (B, T, N, 3)
        pts_2: Tensor of shape (B, T, N, 3)

    Returns:
        Tensor of shape (B, T, N) — angle (in radians) between corresponding vectors.
    """
    # Dot product along last dimension
    cosine_sim = torch.sum(pts_1 * pts_2, dim=-1)  # shape: (B, T, N)

    # Clamp for numerical stability
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

    # Compute angle
    angle = torch.acos(cosine_sim)

    return angle * 180 / torch.pi  # shape: (B, T, N), in degrees


def get_pts_within_from_thresholds(pred_tracks, gt_tracks, use_angles=False, angle_per_pixel=0.2755):
    metrics = {}
    all_pts_within = []
    for thresh in [1, 2, 4, 8, 16]:
        if use_angles:
            is_correct = calc_angular_dist(pred_tracks, gt_tracks) < (thresh * angle_per_pixel)
        else:
            is_correct = torch.sum(torch.square(pred_tracks - gt_tracks), axis=-1) < np.square(thresh)
        metrics["pts_within_" + str(thresh)] = is_correct
        all_pts_within.append(is_correct)
    metrics["average_pts_within"] = torch.mean(torch.stack(all_pts_within, axis=1).float(), axis=1)

    return metrics


def compute_metrics(pred_uv, gt_uv, gt_vis):
    gt_uv = gt_uv.to(pred_uv.device)
    points_within_metrics = get_pts_within_from_thresholds(pred_uv, gt_uv, use_angles=True)
    angular_dists = calc_angular_dist(pred_uv, gt_uv)

    return {"average_pts_within_all": points_within_metrics["average_pts_within"],
            "average_pts_within_in_frame": points_within_metrics["average_pts_within"][gt_vis],
            "average_pts_within_out_of_frame": points_within_metrics["average_pts_within"][~gt_vis],
            "angular_dists_all": angular_dists,
            "angular_dists_in_frame": angular_dists[gt_vis],
            "angular_dists_out_of_frame": angular_dists[~gt_vis]}


def get_average_metrics(metrics, vid_name):
    for metric_name in metrics[vid_name].keys():
        if "avg" not in metrics:
            metrics["avg"] = {}
        metrics["avg"][metric_name] = float(
            torch.concatenate([v[metric_name] for k, v in metrics.items() if k != "avg"]).mean())

    return metrics
