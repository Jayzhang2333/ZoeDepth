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

    if len(knot_values) == 0:
        print('no values')
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
        self.knot_scales = sparse_depth_inv[valid]  / pred_inv[valid]
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


def Anchorinterpolate_knots(map_size, knot_coords, knot_values, interpolate, fill_corners):
    grid_x, grid_y = np.mgrid[0:map_size[0], 0:map_size[1]]

    interpolated_map = griddata(
        points=knot_coords.T,
        values=knot_values,
        xi=(grid_y, grid_x),
        method=interpolate,
        fill_value=1)

    return interpolated_map

class AnchorInterpolator2D(object):
    def __init__(self, sparse_depth, valid):
        self.sparse_depth = sparse_depth
        self.valid = valid

        self.map_size = np.shape(sparse_depth)
        self.num_knots = np.sum(valid)
        nonzero_y_loc = np.nonzero(valid)[0]
        nonzero_x_loc = np.nonzero(valid)[1]
        self.knot_coords = np.stack((nonzero_x_loc, nonzero_y_loc))
        self.knot_values = sparse_depth[valid]

        self.knot_list = []
        for i in range(self.num_knots):
            self.knot_list.append((int(self.knot_coords[0,i]), int(self.knot_coords[1,i])))

        # to be computed
        self.interpolated_map = None
        self.confidence_map = None
        self.output = None

    def generate_interpolated_scale_map(self, interpolate_method, fill_corners=False):
        self.interpolated_scale_map = Anchorinterpolate_knots(
            map_size=self.map_size, 
            knot_coords=self.knot_coords, 
            knot_values=self.knot_values,
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

def compute_scale_ls_torch(prediction, target, mask):
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
    prediction = prediction
    # A = [[a_00, a_01],
    #      [a_10, a_11]]
    # but we only store the needed elements
    # print(f"mask shape: {mask.shape}, relative depth shape {prediction.shape}")
    a_00 = torch.sum(mask * prediction * prediction, dim=sum_dims)
   

    # Right-hand side b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, dim=sum_dims)

    # Prepare output
    x_0 = b_0 / (a_00)
    


    return x_0

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

def estimate_scale_and_shift(estimated: torch.Tensor,
                             ground_truth: torch.Tensor,
                             mask: torch.Tensor):
    """
    Solve for scale >= 0 and shift (unconstrained) that minimize
        sum_{i in valid} (scale*estimated[i] + shift - ground_truth[i])^2
    where 'valid' means mask[i] != 0.

    Parameters
    ----------
    estimated     : (N,1,H,W) Tensor of predicted values
    ground_truth  : (N,1,H,W) Tensor of sparse ground-truth
    mask          : (N,1,H,W) Tensor of 0/1 indicating which pixels are valid

    Returns
    -------
    scale, shift : two scalars (torch Tensors)
        scale >= 0, shift is unconstrained
    """
    # 1) Gather valid data
    #    Shape (num_valid,) after flatten
    valid_est = estimated[mask > 0].view(-1)
    valid_gt  = ground_truth[mask > 0].view(-1)

    n_valid = valid_est.numel()
    if n_valid < 2:
        # Not enough valid data to solve anything; return default
        # or you could return None, None, etc.
        return torch.tensor(1.0, device=estimated.device), torch.tensor(0.0, device=estimated.device)

    # 2) Construct design matrix X = [ [x_0, 1],
    #                                  [x_1, 1],
    #                                  ...
    #                                  [x_{n-1}, 1] ]
    ones = torch.ones_like(valid_est)
    X = torch.stack([valid_est, ones], dim=-1)  # shape (n_valid, 2)

    # 3) Solve the normal equations for [s, t]:
    #       (X^T X) [s, t]^T = X^T y
    #    We'll use torch.linalg.solve for better numerical stability
    #    than explicitly inverting (X^T X).
    xtx = X.transpose(0, 1) @ X      # shape (2,2)
    xty = X.transpose(0, 1) @ valid_gt  # shape (2,)

    # Solve for s, t
    st = torch.linalg.solve(xtx, xty)  # shape (2,)
    s, t = st[0], st[1]

    # 4) Enforce s >= 0
    if s < 0:
        # If unconstrained solution yields negative scale,
        # set s = 0, then the best shift is mean of valid_gt
        s = torch.zeros(1, device=s.device, dtype=s.dtype)
        t = valid_gt.mean()

    return s, t

def compute_scale_and_shift_ls_constrained(prediction, target, mask, eps=1e-6):
    """
    Compute scale (x0) and shift (x1) that map prediction to target using 
    a weighted least-squares estimate under the mask, subject to:
      - x0 >= eps
      - x1 >= 0
    prediction, target, mask: torch.Tensor of shape (b, 1, h, w)
    Returns: (x0, x1) of shape (b,) one per batch element.
    """
    sum_dims = (2, 3)
    
    # Precompute common sums
    a00 = torch.sum(mask * prediction * prediction, dim=sum_dims)
    a01 = torch.sum(mask * prediction, dim=sum_dims)
    a11 = torch.sum(mask, dim=sum_dims)  # also a10 in a symmetric matrix

    b0 = torch.sum(mask * prediction * target, dim=sum_dims)
    b1 = torch.sum(mask * target, dim=sum_dims)

    batch_size = prediction.shape[0]
    x0 = torch.empty(batch_size, device=prediction.device, dtype=prediction.dtype)
    x1 = torch.empty(batch_size, device=prediction.device, dtype=prediction.dtype)
    
    # Precompute the constant term (independent of x0 and x1) if needed for error comparison.
    # Note: Since it does not affect which candidate minimizes the error, we can ignore it.
    #
    # Loop over batch elements (since the optimal candidate may differ per element)
    for i in range(batch_size):
        # If a00 or a11 is nearly zero, fallback to a trivial solution.
        if a00[i] < 1e-8 or a11[i] < 1e-8:
            x0[i] = eps
            x1[i] = 0
            continue

        # Compute unconstrained least-squares solution if possible
        det = a00[i] * a11[i] - a01[i] * a01[i]
        if det > 0:
            x0_u = (a11[i] * b0[i] - a01[i] * b1[i]) / det
            x1_u = (-a01[i] * b0[i] + a00[i] * b1[i]) / det
        else:
            # Fall back if the system is degenerate
            x0_u = eps
            x1_u = 0

        # If the unconstrained solution meets our constraints, use it.
        if x0_u >= eps and x1_u >= 0:
            x0[i] = x0_u
            x1[i] = x1_u
        else:
            # --- Candidate 1: x0 fixed at eps, optimize x1 ---
            # The 1D problem is: minimize sum m*(eps * prediction + x1 - target)^2.
            # Its unconstrained optimum is:
            x1_c1 = (b1[i] - a01[i] * eps) / a11[i]
            # Enforce x1 >= 0.
            x1_c1 = max(x1_c1, 0)
            x0_c1 = eps
            error_c1 = a00[i]*x0_c1**2 + 2*a01[i]*x0_c1*x1_c1 + a11[i]*x1_c1**2 - 2*(b0[i]*x0_c1 + b1[i]*x1_c1)

            # --- Candidate 2: x1 fixed at 0, optimize x0 ---
            # The 1D problem is: minimize sum m*(x0 * prediction - target)^2.
            # Its unconstrained optimum is:
            x0_c2 = b0[i] / a00[i]
            x0_c2 = max(x0_c2, eps)  # enforce x0 >= eps
            x1_c2 = 0
            error_c2 = a00[i]*x0_c2**2 - 2*b0[i]*x0_c2 + a11[i]*x1_c2**2 - 2*b1[i]*x1_c2  # note: a01 term is 0

            # --- Candidate 3: Both constraints active ---
            x0_c3 = eps
            x1_c3 = 0
            error_c3 = a00[i]*x0_c3**2 - 2*b0[i]*x0_c3  # + constant (ignoring terms with x1)

            # Select candidate with minimum error
            errors = [error_c1, error_c2, error_c3]
            candidates = [(x0_c1, x1_c1), (x0_c2, x1_c2), (x0_c3, x1_c3)]
            best_idx = errors.index(min(errors))
            x0[i], x1[i] = candidates[best_idx]

    return x0, x1

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


    def compute_scale(self):
        """
        Calls the LS solver to find the best scale & shift 
        under the mask specified by valid.
        """
        self.scale = compute_scale_ls_torch(
            self.estimate, self.target, self.valid
        )

    def compute_scale_and_shift(self):
        """
        Calls the LS solver to find the best scale & shift 
        under the mask specified by valid.
        """
        self.scale, self.shift = compute_scale_and_shift_ls_torch(
            self.estimate, self.target, self.valid
        )

    def compute_scale_and_shift_constrained(self):
        """
        Calls the LS solver to find the best scale & shift 
        under the mask specified by valid.
        """
        self.scale, self.shift = compute_scale_and_shift_ls_constrained(
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

    def apply_scale(self):
        
        if self.scale is None:
            raise RuntimeError("Must call compute_scale first.")
        # scale, shift: (b,) -> (b,1,1,1) for broadcast
        scale_4d = self.scale.view(-1, 1, 1, 1)
        # print(f'scale is {scale_4d}')
        self.output = self.estimate * scale_4d

    def apply_scale_correction(self, scale_correction_maps):
        self.output = self.output * scale_correction_maps


    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        """
        Clamps the output in the sense described by your original code:
        - If clamp_min is specified (>0), the maximum allowed value is 1/clamp_min
        - If clamp_max is specified, the minimum allowed value is 1/clamp_max
        """
        if self.output is None:
            raise RuntimeError("Must apply scale or scale+shift before clamping.")

        # if clamp_min is not None:
        #     if clamp_min > 0:
        #         clamp_min_inv = 1.0 / clamp_min
        #         self.output[self.output > clamp_min_inv] = clamp_min_inv
        #         # Optional debug check
        #         assert torch.max(self.output) <= clamp_min_inv

        # if clamp_max is not None:
        #     clamp_max_inv = 1.0 / clamp_max
        #     self.output[self.output < clamp_max_inv] = clamp_max_inv
        #     # Optional debug check
        #     assert torch.min(self.output) >= clamp_max_inv

        if clamp_min is not None:
            # If clamp_min is not already a tensor, convert it.
            if not isinstance(clamp_min, torch.Tensor):
                clamp_min = torch.tensor(clamp_min, dtype=torch.float32)
            else:
                clamp_min = clamp_min.to(torch.float32)
            # Use .item() to get the Python scalar for the comparison.
            if clamp_min.item() > 0:
                clamp_min_inv = 1.0 / clamp_min  # This results in a float32 tensor.
                self.output[self.output > clamp_min_inv] = clamp_min_inv
                # Optional debug check
                assert torch.max(self.output) <= clamp_min_inv

        if clamp_max is not None:
            if not isinstance(clamp_max, torch.Tensor):
                clamp_max = torch.tensor(clamp_max, dtype=torch.float32)
            else:
                clamp_max = clamp_max.to(torch.float32)
            clamp_max_inv = 1.0 / clamp_max  # This will be a float32 tensor.
            self.output[self.output < clamp_max_inv] = clamp_max_inv
            # Optional debug check
            assert torch.min(self.output) >= clamp_max_inv