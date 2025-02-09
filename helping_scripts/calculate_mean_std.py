import math
import numpy as np
from PIL import Image
from tqdm import tqdm

def count_lines(filename):
    # Quickly count lines for tqdm
    with open(filename, 'r') as f:
        return sum(1 for _ in f if _.strip())

def compute_mean_std_from_listfile_streaming(list_file_path):
    import math
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    
    total_lines = count_lines(list_file_path)
    
    mean = 0.0
    M2 = 0.0
    count = 0
    
    # Now open the file again and process lines in a streaming fashion
    with open(list_file_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            depth_path = parts[2]

            depth_img = Image.open(depth_path)
            depth_array = np.array(depth_img, dtype=np.uint16)
            depth_array = depth_array.astype(np.float32) / 256.0
            nonzero_depths = depth_array[depth_array > 0]

            for x in nonzero_depths:
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                M2 += delta * delta2

    if count < 2:
        return mean, float('nan')

    variance = M2 / count
    std = math.sqrt(variance)
    return mean, std

if __name__ == "__main__":
    list_txt_file = "train_test_inputs/void_150_combined_training_list_random_filtered.txt"  # replace with your file path
    mean_depth, std_depth = compute_mean_std_from_listfile_streaming(list_txt_file)
    print(f"Global Mean Depth: {mean_depth:.4f}")
    print(f"Global Std Depth:  {std_depth:.4f}")
