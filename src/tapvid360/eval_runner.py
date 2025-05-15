import torch

from tqdm import tqdm

from src.tapvid360.data_utils import get_transform, collate_fn
from src.tapvid360.dataloaders.eval_dataloader import UnitVectorVideoDataset
from src.tapvid360.metrics.tapvid360_metrics import compute_metrics, get_average_metrics
from src.tapvid360.model.cotracker360 import CoTrackerThreeOfflineRotDelta

DS_ROOT = ...  # Parent of the /dataset folder of tapvid360
TRAINED_MODEL_PATH = ... # Path to the checkpoint_last.pth file


def _make_query_points_for_cotracker(gt_unit_vectors, perspective_points):
    init_unit_vector = gt_unit_vectors[:, 0] if gt_unit_vectors.ndim == 4 else gt_unit_vectors
    init_persp_point = perspective_points[:, 0] if perspective_points.ndim == 4 else perspective_points

    return [torch.cat([torch.zeros_like(init_unit_vector[..., 0:1]), init_unit_vector], dim=-1), init_persp_point]


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = UnitVectorVideoDataset(DS_ROOT, "test", get_transform((240, 320)), num_queries=256,
                                     debug_max_items=-1, num_frames=32, no_split=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    model = CoTrackerThreeOfflineRotDelta(stride=4, corr_radius=3, window_len=60, model_resolution=(240, 320)).to(
        device)

    model.load_state_dict(torch.load(TRAINED_MODEL_PATH)["cotracker_rot_delta"])
    model.eval()

    metrics = {}
    for i, sample in enumerate(tqdm(dataloader)):
        sample.to_device(device)
        assert len(sample.video) == 1, "Batch not supported"
        query_points = _make_query_points_for_cotracker(sample.trajectory, sample.query_points)
        coord_preds, vis_preds, confidence_preds, _ = model(sample.video, queries=query_points, mapper=sample.mapper[0])
        metrics[sample.seq_name[0]] = compute_metrics(coord_preds, sample.trajectory, sample.visibility)
        metrics = get_average_metrics(metrics, sample.seq_name[0])
    if len(metrics):
        print(metrics["avg"])
