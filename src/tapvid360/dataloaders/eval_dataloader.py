import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.tapvid360.conversions.mapper import Mappers
from src.tapvid360.data_utils import DataSample


class UnitVectorVideoDataset(Dataset):
    def __init__(self, ds_root, phase, transform=None, debug_max_items=-1, split_filepath=None, split_img_names=None,
                 num_frames=16, num_queries=32, no_split=False):
        super().__init__()
        self.ds_root = Path(ds_root) / "dataset"
        self.phase = phase
        self.transform = transform
        self.spherical_img_names = [f"{item.parent.name}/{item.name}" for sublist in
                                    [list(vid_name.glob("*")) for vid_name in self.ds_root.glob("*")] for item in
                                    sublist]
        self.split_img_names = split_img_names
        if no_split:
            self.split_img_names = self.spherical_img_names
        if self.split_img_names is None and not no_split:
            if split_filepath is not None:
                self.split_file = self.ds_root / "splits_new.json"
                with open(self.split_file, "r") as f:
                    self.split_img_names = json.load(f)[self.phase]
            else:
                # Just get a 90/10 split for train and test
                split = int(len(self.spherical_img_names) * 0.9)  # 90% for training
                if phase == "train":
                    self.split_img_names = self.spherical_img_names[:split]
                else:
                    self.split_img_names = self.spherical_img_names[split:]
        if debug_max_items > 0:
            if debug_max_items == 0:
                raise ValueError("debug_max_items must be greater than 1")
            self.split_img_names = self.split_img_names[:debug_max_items]
        self.num_frames = num_frames
        self.num_queries = num_queries

    def _get_the_data(self, spherical_img_name):
        t_imgs = []
        persp_img_shapes = []
        for i in range(self.num_frames):
            persp_img = Image.open(self.ds_root / spherical_img_name / "perspective_frames" / f"{i}.jpg")
            persp_img_shapes.append(persp_img.size)
            t_imgs.append(self.transform(persp_img))
        t_imgs = torch.stack(t_imgs)

        data = torch.load(self.ds_root / spherical_img_name / "data.pt", weights_only=False)
        _, _, w, h = t_imgs.shape
        mapper = Mappers(w, h, data["equi_w"], data["equi_h"], fov_x=data["fov_x"])

        return DataSample(
            video=t_imgs[:self.num_frames],
            trajectory=data["unit_vectors"][:self.num_frames, :self.num_queries],
            query_points=data["persp_points"][0][:self.num_frames, :self.num_queries],
            mapper=mapper,
            seq_name=spherical_img_name,
            rotations=data["rotations"][:self.num_frames, :self.num_queries],
            orig_image_shapes=persp_img_shapes,
            visibility=data["persp_points_vis"][0][:self.num_frames,
                       :self.num_queries] if "persp_points_vis" in data else None,
        )

    def __len__(self):
        return len(self.split_img_names)

    def __getitem__(self, idx):
        return self._get_the_data(self.split_img_names[idx])
