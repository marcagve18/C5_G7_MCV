
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set the path to the dataset directory
data_dir = "/home/mcv/datasets/C5/KITTI-MOTS/instances"


instance_dirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
print(f"Found {len(instance_dirs)} instance directories.")

instance_frame_counts = {}
for instance in instance_dirs:
    instance_number = int(os.path.basename(instance))
    # Assuming every file in the instance folder is a frame
    frame_files = sorted(glob.glob(os.path.join(instance, "*")))
    instance_frame_counts[instance_number] = len(frame_files)

keys = list(instance_frame_counts.keys())
values = list(instance_frame_counts.values())

plt.figure(figsize=(10,6))
plt.bar(keys, values, width=0.8)
plt.xlabel('Instance number')
plt.ylabel('Frames')
plt.title('Frame distribution')
plt.xticks(keys)
plt.tight_layout()
plt.savefig("frame_count_distribution.png")


## Data splitting

sorted_items = sorted(instance_frame_counts.items(), key=lambda x: x[1], reverse=True)

# Total sum of all values
total_sum = sum(values)
target_80 = total_sum * 0.78 # Optimal value, found by trying from range 75 to 80

# Find the best split
current_sum = 0
split_index = 0

for i, (key, value) in enumerate(sorted_items):
    current_sum += value
    if current_sum >= target_80:
        split_index = i
        break

# Split data
data_80 = dict(sorted_items[:split_index+1])
data_20 = dict(sorted_items[split_index+1:])

# Display the results
data_80_sum = sum(data_80.values())
data_20_sum = sum(data_20.values())

print(f"Train split instances:  {list(data_80.keys())} | {data_80_sum} | {data_80_sum/(data_80_sum+data_20_sum)}")
print(f"Test split instances:  {list(data_20.keys())} | {data_20_sum} | {data_20_sum/(data_80_sum+data_20_sum)}")

print(data_80_sum+data_20_sum)