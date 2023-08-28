import matplotlib.pyplot as plt
import numpy as np
from yupi.graphics import plot_2d, plot_hist

from pactus import Dataset

# Data Loading
ds = Dataset.hurdat2()
print(f"Loaded dataset: {ds.name}")
print(f"Total trajectories: {len(ds.trajs)}")
print(f"Different classes: {ds.classes}")

# Inspecting a single trajectory
traj_idx = 20
traj, label = ds.trajs[traj_idx], ds.labels[traj_idx]
plot_2d([traj], legend=False, show=False)
plt.legend([f"Label: {label}"])
plt.title(f"Trajectory no. {traj_idx}")
plt.xlabel("lon")
plt.ylabel("lat")
plt.show()

# Inspecting a subset of the first trajectories
traj_count = 200
first_trajs = ds.trajs[:traj_count]
plot_2d(first_trajs, legend=False, color="#2288dd", show=False)
plt.title(f"First {traj_count} trajectories")
plt.xlabel("lon")
plt.ylabel("lat")
plt.show()

# Inspecting the distribution of trajectories on each class
plt.bar(ds.label_counts.keys(), ds.label_counts.values())
plt.title("Trajectory count by class")
plt.xlabel("Class")
plt.show()

# Inspecting the lenght distribution of the trajectories in the dataset
lengths = np.array([len(traj) for traj in ds.trajs])
plot_hist(lengths, bins=40, show=False)
plt.title("Trajectory lengths historgram")
plt.xlabel("Length")
plt.show()