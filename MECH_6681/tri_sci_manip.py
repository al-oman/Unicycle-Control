import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as smp
from matplotlib import animation
from matplotlib.animation import PillowWriter
import csv

M_const = 1.0
R_const = 1.0
l_const = 1.0
mu_const = 1.0
g_const = 9.8

# Read the CSV file
data = []
with open('/Users/alex/Documents/VSC_Python/MECH_6681/output.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)


# Convert data to numpy array
data = np.array(data, dtype=float)

x_d = -1*R_const*data[:,8]*np.cos(data[:,3])
y_d = -1*R_const*data[:,8]*np.sin(data[:,3])
theta = data[:,1]
phi = data[:,3]

t = np.linspace(0, 10, 1001)



# Calculate x and y values for each time step
x_cont = np.empty_like(t)
y_cont = np.empty_like(t)

for i, time in enumerate(t):
    x_cont[i] = np.trapz(x_d[:i+1], x=None, dx=0.01)
    y_cont[i] = np.trapz(y_d[:i+1], x=None, dx=0.01)
    
x_disk = x_cont - R_const*np.sin(theta)*np.sin(phi)
y_disk = y_cont + R_const*np.sin(theta)*np.cos(phi)

x_rotor = x_disk - l_const*np.sin(theta)*np.sin(phi)
y_rotor = y_disk + l_const*np.sin(theta)*np.cos(phi)

z_cont = np.empty_like(t)
z_disk = R_const*np.cos(theta)
z_rotor = z_disk + l_const*np.cos(theta)
    
# Add x_values, y_values, xd_values, yd_values as columns to data
data = np.column_stack((data, x_cont, x_d, y_cont, y_d, x_rotor, y_rotor, z_rotor))

# Write the manipulated data to a CSV file
output_file = '/Users/alex/Documents/VSC_Python/MECH_6681/manipulated_data.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
print(f"Manipulated data has been written to {output_file}")

# Plot the data
plt.plot(x_rotor, y_rotor)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Position vs Time')
plt.legend(['X', 'Y'])
plt.show()
