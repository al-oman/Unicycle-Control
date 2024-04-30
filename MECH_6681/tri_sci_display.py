import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import csv

data = []
with open('/Users/alex/Documents/VSC_Python/MECH_6681/manipulated_data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
        
data = np.array(data, dtype=float)
num_frames = len(data)

t = data[:,0]
x_R = data[:,13]
y_R = data[:,14]
z_R = data[:,15]


coords = np.column_stack((x_R, y_R, z_R))

def update_graph(num):
    graph._offsets3d = (coords[:,0], coords[:,1], coords[:,2])

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(0, 3)

graph = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=t, cmap='viridis', s=10)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, 
                               interval=40, blit=False)

plt.show()