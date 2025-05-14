import heapq
import argparse

# Helper functions for the running median (using two heaps)
def add_number(num, lowers, highers):
    if not lowers or num < -lowers[0]:
        heapq.heappush(lowers, -num)  # max heap (store negative values)
    else:
        heapq.heappush(highers, num)   # min heap
    # Rebalance heaps if necessary
    if len(lowers) > len(highers) + 1:
        heapq.heappush(highers, -heapq.heappop(lowers))
    elif len(highers) > len(lowers) + 1:
        heapq.heappush(lowers, -heapq.heappop(highers))

def get_median(lowers, highers):
    if len(lowers) == len(highers):
        return (-lowers[0] + highers[0]) / 2.0
    elif len(lowers) > len(highers):
        return float(-lowers[0])
    else:
        return float(highers[0])

# Command-line argument parsing
parser = argparse.ArgumentParser(
    description="Compute dataset statistics with optional filtering on depth values."
)
parser.add_argument(
    '--master_file', type=str, default='./train_test_inputs/flease_testing/flsea_preserved_testing.txt',
    help='Path to the master file containing paths to feature files.'
)
parser.add_argument(
    '--filter_mode', type=str, choices=['none', 'portion', 'range'], default='none',
    help="Filtering mode: 'none' (use all points), 'portion' (keep a fraction from one end), or 'range' (keep points within a depth range)."
)
# Options for portion-based filtering
parser.add_argument(
    '--filter_side', type=str, choices=['lower', 'upper'], default='lower',
    help='For portion filtering, choose which end of the sorted depths to keep ("lower" or "upper").'
)
parser.add_argument(
    '--filter_fraction', type=float, default=1/3,
    help='For portion filtering, fraction of points to keep (e.g., 0.33 for one-third, 0.25 for one-fourth).'
)
# Options for range-based filtering
parser.add_argument(
    '--min_depth', type=float, default=None,
    help='For range filtering, minimum depth value to include.'
)
parser.add_argument(
    '--max_depth', type=float, default=None,
    help='For range filtering, maximum depth value to include.'
)

args = parser.parse_args()

# User options
master_file_path = args.master_file
filter_mode = args.filter_mode

# Initialize overall dataset statistics variables
points_count_list = []   # (Filtered) number of points per image
total_points = 0
total_depth_sum = 0.0

# Lists to record per-image maximum and minimum depth (after filtering)
image_max_list = []
image_min_list = []

# Heaps for computing the running median across all (filtered) depth values
lowers = []  # max heap (store negative values)
highers = []  # min heap

# Process the master file (each row contains three paths; we use the third one)
with open(master_file_path, 'r') as master_file:
    for line in master_file:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # Skip malformed lines
        third_path = parts[2]
        
        try:
            with open(third_path, 'r') as data_file:
                # Skip header line (assumed to be "row column depth")
                header = data_file.readline()
                depths = []
                # Read each line from the feature file
                for row in data_file:
                    row = row.strip()
                    if not row:
                        continue
                    # Supports either comma-delimited or space-delimited values
                    tokens = row.split(',') if ',' in row else row.split()
                    if len(tokens) < 3:
                        continue
                    try:
                        depth = float(tokens[2])
                    except ValueError:
                        continue  # Skip rows with invalid depth values
                    depths.append(depth)
                
                # Apply filtering based on selected mode
                if filter_mode == 'portion' and depths:
                    depths.sort()
                    cutoff = int(len(depths) * args.filter_fraction)
                    cutoff = max(cutoff, 1)  # Ensure at least one point is kept
                    if args.filter_side == 'lower':
                        filtered_depths = depths[:cutoff]
                    else:
                        filtered_depths = depths[-cutoff:]
                elif filter_mode == 'range' and depths:
                    # Keep points that fall within the specified depth range
                    filtered_depths = [
                        d for d in depths
                        if (args.min_depth is None or d >= args.min_depth) and
                           (args.max_depth is None or d <= args.max_depth)
                    ]
                else:
                    # 'none' or if depths is empty: use all points
                    filtered_depths = depths
                
                # Record the (filtered) number of points for this image
                image_point_count = len(filtered_depths)
                points_count_list.append(image_point_count)
                
                # Record per-image max and min from filtered depths
                if filtered_depths:
                    image_max_list.append(max(filtered_depths))
                    image_min_list.append(min(filtered_depths))
                
                # Update overall totals and the running median using the filtered points
                for d in filtered_depths:
                    total_depth_sum += d
                    total_points += 1
                    add_number(d, lowers, highers)
        except Exception as e:
            print(f"Error reading file {third_path}: {e}")

# Compute overall statistics

# 1. (Filtered) point counts per image
num_images = len(points_count_list)
avg_points = sum(points_count_list) / num_images if num_images > 0 else 0
points_count_list.sort()
if num_images > 0:
    if num_images % 2 == 1:
        median_points = points_count_list[num_images // 2]
    else:
        median_points = (points_count_list[num_images // 2 - 1] + points_count_list[num_images // 2]) / 2
else:
    median_points = 0

# 2. Overall depth statistics (filtered points)
avg_depth = total_depth_sum / total_points if total_points > 0 else 0
median_depth = get_median(lowers, highers)

# 3. Dataset-level statistics for per-image maximum depths
if image_max_list:
    avg_max = sum(image_max_list) / len(image_max_list)
    image_max_list.sort()
    n_max = len(image_max_list)
    if n_max % 2 == 1:
        median_max = image_max_list[n_max // 2]
    else:
        median_max = (image_max_list[n_max // 2 - 1] + image_max_list[n_max // 2]) / 2
else:
    avg_max = median_max = None

# 4. Dataset-level statistics for per-image minimum depths
if image_min_list:
    avg_min = sum(image_min_list) / len(image_min_list)
    image_min_list.sort()
    n_min = len(image_min_list)
    if n_min % 2 == 1:
        median_min = image_min_list[n_min // 2]
    else:
        median_min = (image_min_list[n_min // 2 - 1] + image_min_list[n_min // 2]) / 2
else:
    avg_min = median_min = None

# Output the computed statistics
print("Average number of (filtered) points per image:", avg_points)
print("Median number of (filtered) points per image:", median_points)
print("Average depth (filtered points):", avg_depth)
print("Median depth (filtered points):", median_depth)
print("Average max depth (per image):", avg_max)
print("Median max depth (per image):", median_max)
print("Average min depth (per image):", avg_min)
print("Median min depth (per image):", median_min)
