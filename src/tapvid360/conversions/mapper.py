from typing import Optional

import torch


def get_pixel_focal_length(crop_width: int, crop_height: int, fov_x: int, fov_y: Optional[int] = None):
    if fov_y is None:
        fov_y = float(crop_height) / crop_width * fov_x
    # Convert FOV from degrees to radians
    fov_x_rad = torch.deg2rad(torch.tensor(fov_x))
    fov_y_rad = torch.deg2rad(torch.tensor(fov_y))
    # Calculate the focal lengths in pixels
    f_x = crop_width / (2 * torch.tan(fov_x_rad / 2))
    f_y = crop_height / (2 * torch.tan(fov_y_rad / 2))

    return f_x, f_y


class Mappers:
    def __init__(self, crop_width: int, crop_height: int, equirectangular_width: int, equirectangular_height: int,
                 fov_x: float = None):
        """
        Args:

            crop_width (int): The width of the perspective image.
            crop_height (int): The height of the perspective image.
            equirectangular_width (int): The width of the equirectangular image.
            equirectangular_height (int): The height of the equirectangular image.
            fox_x (Optional[float]): The horizontal field of view in degrees. If None, it will be calculated from f_x.
        """

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.equirectangular_width = equirectangular_width
        self.equirectangular_height = equirectangular_height

        self.fov_x = fov_x
        self.fov_y = float(self.crop_height) / self.crop_width * self.fov_x
        self.f_x, self.f_y = get_pixel_focal_length(self.crop_width, self.crop_height, self.fov_x, fov_y=self.fov_y)

    def camera_coords_to_perspective_point(self, v: torch.Tensor, set_out_of_bounds_to_nan=True):
        """
        Converts a normalised 3D direction vector in the perspective camera's local coordinate system
        back into a 2D pixel coordinate in the perspective image.

        Args:
            v (torch.Tensor): A normalised 3D direction vector (batch_size, 3) in the camera's local coordinate system.
            set_out_of_bounds_to_nan (bool): If True, sets out-of-bounds pixel coordinates to NaN. Defaults to True.

        Returns:
            torch.Tensor: Perspective image row (i) and column (j) of shape (n_frames, n_points, 3).
            in_bounds (torch.Tensor): A boolean tensor indicating if the vector is within the FOV.
        """
        # Extract components
        v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]

        # First lets test if the point will be in bounds of the perspective image FOV
        # Compute azimuth (phi) and elevation (theta) angles
        phi = torch.atan2(v_y, v_x)  # Azimuth
        theta = torch.asin(v_z)  # Elevation

        # Compute the FOV bounds in radians
        half_fov_x = torch.deg2rad(torch.tensor(self.fov_x / 2))
        half_fov_y = torch.deg2rad(torch.tensor(self.fov_y / 2))

        # Check if phi and theta are within the FOV
        in_bounds = (-half_fov_x <= phi) & (phi <= half_fov_x) & \
                    (-half_fov_y <= theta) & (theta <= half_fov_y)

        # Need to divide by the v_x here as it was normalised previously
        i = self.f_x * (v_y / v_x) + self.crop_width / 2
        j = self.f_y * (-v_z / v_x) + self.crop_height / 2

        if set_out_of_bounds_to_nan:
            i[~in_bounds] = torch.nan
            j[~in_bounds] = torch.nan

        return torch.stack([i, j], dim=-1), in_bounds
