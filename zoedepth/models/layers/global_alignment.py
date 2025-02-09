import numpy as np
np.set_printoptions(suppress=True)

from scipy.interpolate import griddata
import torch

def interpolate_knots(map_size, knot_coords, knot_values, interpolate, fill_corners):
    grid_x, grid_y = np.mgrid[0:map_size[0], 0:map_size[1]]

    interpolated_map = griddata(
        points=knot_coords.T,
        values=knot_values,
        xi=(grid_y, grid_x),
        method=interpolate,
        fill_value=1) 

    return interpolated_map

class Interpolator2D(object):
    def __init__(self, pred, sparse_depth, valid):
        self.pred = pred
        self.sparse_depth = sparse_depth
        self.valid = valid

        self.map_size = np.shape(pred)
        self.num_knots = np.sum(valid)
        nonzero_y_loc = np.nonzero(valid)[0]
        nonzero_x_loc = np.nonzero(valid)[1]
        self.knot_coords = np.stack((nonzero_x_loc, nonzero_y_loc))
        self.knot_shifts = sparse_depth[valid]# - pred[valid]

        self.knot_list = []
        for i in range(self.num_knots):
            self.knot_list.append((int(self.knot_coords[0,i]), int(self.knot_coords[1,i])))

        # to be computed
        self.interpolated_map = None
        self.confidence_map = None
        self.output = None

    def generate_interpolated_residual_map(self, interpolate_method, fill_corners=False):
        self.interpolated_residual_map = interpolate_knots(
            map_size=self.map_size, 
            knot_coords=self.knot_coords, 
            knot_values=self.knot_shifts,
            interpolate=interpolate_method,
            fill_corners=fill_corners
        ).astype(np.float32)

def interpolate_knots_original(map_size, knot_coords, knot_values, interpolate, fill_corners):
    grid_x, grid_y = np.mgrid[0:map_size[0], 0:map_size[1]]

    interpolated_map = griddata(
        points=knot_coords.T,
        values=knot_values,
        xi=(grid_y, grid_x),
        method=interpolate,
        fill_value=1.0)

    return interpolated_map

class Interpolator2D_original(object):
    def __init__(self, pred_inv, sparse_depth_inv, valid):
        self.pred_inv = pred_inv
        self.sparse_depth_inv = sparse_depth_inv
        self.valid = valid

        self.map_size = np.shape(pred_inv)
        self.num_knots = np.sum(valid)
        nonzero_y_loc = np.nonzero(valid)[0]
        nonzero_x_loc = np.nonzero(valid)[1]
        self.knot_coords = np.stack((nonzero_x_loc, nonzero_y_loc))
        self.knot_scales = sparse_depth_inv[valid] / pred_inv[valid]
        self.knot_shifts = sparse_depth_inv[valid] - pred_inv[valid]

        self.knot_list = []
        for i in range(self.num_knots):
            self.knot_list.append((int(self.knot_coords[0,i]), int(self.knot_coords[1,i])))

        # to be computed
        self.interpolated_map = None
        self.confidence_map = None
        self.output = None

    def generate_interpolated_scale_map(self, interpolate_method, fill_corners=False):
        self.interpolated_scale_map = interpolate_knots_original(
            map_size=self.map_size, 
            knot_coords=self.knot_coords, 
            knot_values=self.knot_scales,
            interpolate=interpolate_method,
            fill_corners=fill_corners
        ).astype(np.float32)


# class Interpolator2D(object):
#     # def __init__(self, pred_inv, sparse_depth_inv, valid):
#     def __init__(self, sparse_depth_inv, valid):
#         # self.pred_inv = pred_inv
#         self.sparse_depth_inv = sparse_depth_inv
#         self.valid = valid

#         self.map_size = np.shape(sparse_depth_inv)
#         self.num_knots = np.sum(valid)
#         nonzero_y_loc = np.nonzero(valid)[0]
#         nonzero_x_loc = np.nonzero(valid)[1]
#         self.knot_coords = np.stack((nonzero_x_loc, nonzero_y_loc))
#         # self.knot_scales = sparse_depth_inv[valid] / pred_inv[valid]
#         self.knot_scales = sparse_depth_inv[valid]
#         # self.knot_shifts = sparse_depth_inv[valid] - pred_inv[valid]

#         self.knot_list = []
#         for i in range(self.num_knots):
#             self.knot_list.append((int(self.knot_coords[0,i]), int(self.knot_coords[1,i])))

#         # to be computed
#         self.interpolated_map = None
#         self.confidence_map = None
#         self.output = None

#     def generate_interpolated_scale_map(self, interpolate_method, fill_corners=False):
#         self.interpolated_scale_map = interpolate_knots(
#             map_size=self.map_size, 
#             knot_coords=self.knot_coords, 
#             knot_values=self.knot_scales,
#             interpolate=interpolate_method,
#             fill_corners=fill_corners
#         ).astype(np.float32)


def compute_scale(prediction, target, mask):
    # tuple specifying with axes to sum
    sum_axes = (0, 1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, sum_axes)
    
    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, sum_axes)
    

    x_0 = b_0 / a_00
    # print(x_0)
    return x_0

def compute_scale_and_shift_ls(prediction, target, mask):
    # tuple specifying with axes to sum
    sum_axes = (0, 1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, sum_axes)
    a_01 = np.sum(mask * prediction, sum_axes)
    a_11 = np.sum(mask, sum_axes)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, sum_axes)
    b_1 = np.sum(mask * target, sum_axes)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # print(det)
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    # print(f"\n a_00 {a_00}, a_01 {a_01}, a_11 {a_11}, b_0 {b_0}, b_1 {b_1}")
    return x_0, x_1



class LeastSquaresEstimator(object):
    def __init__(self, estimate, target, valid):
        self.estimate = estimate
        self.target = target
        self.valid = valid

        # to be computed
        self.scale = 1.0
        self.shift = 0.0
        self.output = None

    def compute_scale_and_shift(self):
        self.scale, self.shift = compute_scale_and_shift_ls(self.estimate, self.target, self.valid)
        # print(f"original scale {self.scale} and shift {self.shift}")

    def compute_scale(self):
        self.scale = compute_scale(self.estimate, self.target, self.valid)
        # print(f"original scale {self.scale} and shift {self.shift}")

    def apply_scale_and_shift(self):
        self.output = self.estimate * self.scale + self.shift
        self.output_no_shift = self.estimate * self.scale #+ self.shift

    def apply_scale(self):
        self.output = self.estimate * self.scale #+ self.shift
        # self.output_no_shift = self.estimate * self.scale #+ self.shift

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min > 0:
                clamp_min_inv = np.float32(1.0) / np.float32(clamp_min)
                self.output[self.output > clamp_min_inv] = np.float32(clamp_min_inv)
                assert np.max(self.output) <= clamp_min_inv
            else:  # divide by zero, so skip
                pass
        if clamp_max is not None:
            clamp_max_inv = np.float32(1.0) / np.float32(clamp_max)
            self.output[self.output < clamp_max_inv] = np.float32(clamp_max_inv)
            # if np.min(self.output) < clamp_max_inv:
            #     print(type(np.min(self.output)))
            #     print(type(clamp_max_inv))
            assert np.min(self.output) >= np.float32(clamp_max_inv)
        # check for nonzero range
        # assert np.min(self.output) != np.max(self.output)

def compute_scale_and_shift_ls_torch(prediction, target, mask):
    """
    Compute scale and shift that map prediction to target 
    using a least-squares estimate, under the 'mask' region.

    prediction, target, mask: torch.Tensor of shape (b, 1, h, w)
    Returns: (scale, shift) of shape (b,)
             one scale and shift per batch element.
    """

    # We sum over the spatial dimensions (h, w), keeping batch dimension separate.
    # This yields one scale and shift value per batch entry.
    sum_dims = (2, 3)  # sum over h and w

    # A = [[a_00, a_01],
    #      [a_10, a_11]]
    # but we only store the needed elements
    # print(f"mask shape: {mask.shape}, relative depth shape {prediction.shape}")
    a_00 = torch.sum(mask * prediction * prediction, dim=sum_dims)
    a_01 = torch.sum(mask * prediction, dim=sum_dims)
    a_11 = torch.sum(mask, dim=sum_dims)  # same as a_10 in your original approach

    # Right-hand side b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, dim=sum_dims)
    b_1 = torch.sum(mask * target, dim=sum_dims)

    # Prepare output
    x_0 = torch.zeros_like(b_0)  # scale
    x_1 = torch.zeros_like(b_1)  # shift

    # Determinant: det = a_00 * a_11 - (a_01)^2
    det = a_00 * a_11 - a_01 * a_01

    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class LeastSquaresEstimatorTorch:
    """
    A PyTorch-based least-squares estimator that computes a scale and shift 
    to best map 'estimate' to 'target' in the region where 'valid'=1.
    
    Assumes estimate, target, valid have shape (b, 1, h, w).
    You get one scale and shift per batch entry.
    """
    def __init__(self, estimate: torch.Tensor, 
                 target: torch.Tensor, 
                 valid: torch.Tensor):
        """
        estimate, target, valid must all be shape (b, 1, h, w) or broadcastable to that.
        """
        self.estimate = estimate
        self.target = target
        self.valid = valid

        # To be computed
        self.scale = None  # shape (b,)
        self.shift = None  # shape (b,)
        self.output = None

    def compute_scale_and_shift(self):
        """
        Calls the LS solver to find the best scale & shift 
        under the mask specified by valid.
        """
        self.scale, self.shift = compute_scale_and_shift_ls_torch(
            self.estimate, self.target, self.valid
        )

    def apply_scale_and_shift(self):
        """
        Applies the computed scale and shift to self.estimate
        and stores the result in self.output.
        
        Since scale, shift each have shape (b,), we need to
        broadcast them across (h, w).
        """
        if self.scale is None or self.shift is None:
            raise RuntimeError("Must call compute_scale_and_shift first.")
        # scale, shift: (b,) -> (b,1,1,1) for broadcast
        scale_4d = self.scale.view(-1, 1, 1, 1)
        shift_4d = self.shift.view(-1, 1, 1, 1)

        self.output = self.estimate * scale_4d + shift_4d


    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        """
        Clamps the output in the sense described by your original code:
        - If clamp_min is specified (>0), the maximum allowed value is 1/clamp_min
        - If clamp_max is specified, the minimum allowed value is 1/clamp_max
        """
        if self.output is None:
            raise RuntimeError("Must apply scale or scale+shift before clamping.")

        if clamp_min is not None:
            if clamp_min > 0:
                clamp_min_inv = 1.0 / clamp_min
                self.output[self.output > clamp_min_inv] = clamp_min_inv
                # Optional debug check
                assert torch.max(self.output) <= clamp_min_inv

        if clamp_max is not None:
            clamp_max_inv = 1.0 / clamp_max
            self.output[self.output < clamp_max_inv] = clamp_max_inv
            # Optional debug check
            assert torch.min(self.output) >= clamp_max_inv