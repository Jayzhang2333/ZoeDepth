import heapq

# Running median helper functions using two heaps
def add_number(num, lowers, highers):
    if not lowers or num < -lowers[0]:
        heapq.heappush(lowers, -num)  # max heap (store negative values)
    else:
        heapq.heappush(highers, num)  # min heap

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

# Initialize variables for overall dataset statistics
points_count_list = []  # to store the filtered number of points for each image
total_points = 0
total_depth_sum = 0.0

# Heaps for computing the running median of depth values over the entire dataset
lowers = []  # max heap (store negative values)
highers = []  # min heap

# Process the master file (replace 'master.txt' with your actual master file name)
with open('./train_test_inputs/lizard/lizard_image_depth_paths_subsampled_depth.txt', 'r') as master_file:
    for line in master_file:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # Skip malformed lines
        third_path = parts[2]
        
        try:
            with open(third_path, 'r') as data_file:
                # Read and skip the header line (assumed to be "row column depth")
                header = data_file.readline()
                # Collect all depth values for this image in a list
                depths = []
                for row in data_file:
                    row = row.strip()
                    if not row:
                        continue
                    # Support either space-delimited or comma-delimited values
                    tokens = row.split(',') if ',' in row else row.split()
                    if len(tokens) < 3:
                        continue
                    try:
                        depth = float(tokens[2])
                    except ValueError:
                        continue  # Skip rows with invalid depth values
                    depths.append(depth)
                
                # Only consider the lower 1/3 of points (points with the smallest depth values)
                if depths:
                    depths.sort()
                    cutoff = len(depths) // 3  # integer division
                    filtered_depths = depths[:cutoff] if cutoff > 0 else depths[:1]  # ensure at least one point if available
                else:
                    filtered_depths = []

                # Update per-image point count (filtered points)
                image_point_count = len(filtered_depths)
                points_count_list.append(image_point_count)
                
                # Update overall totals and running median with the filtered points
                for d in filtered_depths:
                    total_depth_sum += d
                    total_points += 1
                    add_number(d, lowers, highers)
        except Exception as e:
            print(f"Error reading file {third_path}: {e}")

# Calculate overall statistics

# Average and median number of points per image (after filtering)
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

# Average depth (filtered points)
avg_depth = total_depth_sum / total_points if total_points > 0 else 0

# Median depth using the running median heaps
median_depth = get_median(lowers, highers)

print("Average number of filtered points per image:", avg_points)
print("Median number of filtered points per image:", median_points)
print("Average depth (filtered points):", avg_depth)
print("Median depth (filtered points):", median_depth)
