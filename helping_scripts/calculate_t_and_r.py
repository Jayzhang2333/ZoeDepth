import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_relative_pose(t1, roll1, pitch1, yaw1, vel1, 
                          t2, roll2, pitch2, yaw2, vel2, method='average'):
    """
    Compute the relative pose transformation (rotation and translation)
    between two camera poses.

    Parameters:
      t1, t2: Timestamps (seconds) for pose1 and pose2.
      roll1, pitch1, yaw1: Euler angles for pose1 in degrees (roll, pitch, yaw).
      roll2, pitch2, yaw2: Euler angles for pose2 in degrees (roll, pitch, yaw).
      vel1, vel2: Velocities [vx, vy, vz] in m/s (assumed to be in the camera's local frame).
      method: 'average' or 'pose1'. Use 'average' to average the velocities of both poses,
              or 'pose1' to use the velocity at the first pose only.

    Returns:
      R_relative: 3x3 numpy array representing the relative rotation matrix (from pose1 to pose2).
      t_relative: 3-element numpy array representing the relative translation vector in pose1's coordinate frame.
    """
    # Compute time difference
    dt = t2 - t1
    
    # Convert Euler angles to rotation matrices (assuming roll, pitch, yaw corresponds to 'xyz' order)
    R1 = R.from_euler('xyz', [roll1, pitch1, yaw1], degrees=True).as_matrix()
    R2 = R.from_euler('xyz', [roll2, pitch2, yaw2], degrees=True).as_matrix()
    
    # Compute the relative rotation: from pose1 to pose2
    R_relative = R2 @ R1.T
    
    # Convert velocity lists to numpy arrays
    vel1 = np.array(vel1)
    vel2 = np.array(vel2)
    
    # Transform velocities from the camera's local frame to the world frame
    world_vel1 = R1 @ vel1
    world_vel2 = R2 @ vel2
    
    # Compute the translation by integrating velocity over dt.
    # Here, we choose to average the two velocities.
    if method == 'average':
        avg_world_vel = (world_vel1 + world_vel2) / 2.0
        translation_world = avg_world_vel * dt
    elif method == 'pose1':
        translation_world = world_vel1 * dt
    else:
        raise ValueError("Invalid method specified. Use 'average' or 'pose1'.")
    
    # Express the translation in the coordinate frame of the first pose:
    t_relative = R1.T @ translation_world
    
    return R_relative, t_relative

# Example usage:
if __name__ == "__main__":
    # Pose 1 inputs
    t1 = 37.40824
    yaw1, pitch1, roll1 = -2.82440090179443,-0.0879656225442886,-0.0159724280238152       # in degrees
    vel1 =  [0.047716581671988,0.134311142267751,0.104402057146777]                         # in m/s (camera local frame)
   
    # Pose 2 inputs (after 1 second)
    t2 = 37.94248
    yaw2, pitch2, roll2 = -2.81407284736633,-0.0775604024529457,-0.0184043571352959        # degrees (example: small yaw change)
    vel2 = [0.0321258379897244,0.123785899774247,0.0933871607380904]                       # in m/s (camera local frame)
    
    # Compute relative transformation (using average velocity)
    R_rel, t_rel = compute_relative_pose(t1, roll1, pitch1, yaw1, vel1,
                                         t2, roll2, pitch2, yaw2, vel2, method='average')
    
    print("Relative Rotation Matrix (from pose1 to pose2):")
    print(R_rel)
    print("\nRelative Translation Vector (in pose1's coordinate frame):")
    print(t_rel)
