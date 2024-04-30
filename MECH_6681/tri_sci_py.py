import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import sympy as smp
# from matplotlib import animation
# from matplotlib.animation import PillowWriter
import csv
# import pickle
import dill

READ_MODE = False

#Set the values of the constants (for code use)
M_const = 1.0
R_const = 1.0
l_const = 1.0
mu_const = 1.0
g_const = 9.8

#Initial Conditions
theta_0 = 0.5
theta_d_0 = 0.0
phi_0 = 0.0
phi_d_0 = 0.0
chi_0 = 0.0
chi_d_0 = 0.0
psi_0 = 0.0
psi_d_0 = 2.0
x_0 = 0.0
x_d_0 = -1*R_const*psi_d_0*np.cos(phi_0)
y_0 = 0.0
y_d_0 = -1*R_const*psi_d_0*np.sin(phi_0)
theta_0 = float(theta_0)
theta_d_0 = float(theta_d_0)
phi_0 = float(phi_0)
phi_d_0 = float(phi_d_0)
chi_0 = float(chi_0)
chi_d_0 = float(chi_d_0)
psi_0 = float(psi_0)
psi_d_0 = float(psi_d_0)
x_0 = float(x_0)
y_0 = float(y_0)


#Define the symbols for variables
theta, phi, chi, psi, x, y, = smp.symbols('theta phi chi psi x y', cls=smp.Function)
#Define the symbols for constants
A_d, B_d, A_r, B_r, M, R, l, mu, g, t = smp.symbols('A_d B_d A_r B_r M R l mu g t')

#Create Functions for the variables
theta = theta(t)
phi = phi(t)
chi = chi(t)
psi = psi(t)
x = x(t)
y = y(t)

#Define the 1st and 2nd derivatives of the variables
theta_d = smp.diff(theta, t)
phi_d = smp.diff(phi, t)
chi_d = smp.diff(chi, t)
psi_d = smp.diff(psi, t)
x_d = smp.diff(x, t)
y_d = smp.diff(y, t)

theta_dd = smp.diff(theta_d, t)
phi_dd = smp.diff(phi_d, t)
chi_dd = smp.diff(chi_d, t)
psi_dd = smp.diff(psi_d, t)
x_dd = smp.diff(x_d, t)
y_dd = smp.diff(y_d, t)
print("done defining variables and derivatives")

x_d = -1*R*psi_d*smp.cos(phi)
y_d = -1*R*psi_d*smp.sin(phi)

if(not READ_MODE):
        #Define Variables in the Simplified Lagrange Equation: 
        # L = Kd + Kr + 0.5M*v_M^2 + 0.5mu*v_mu^2 - V
        K_d = smp.Rational(1,2)*A_d*(theta_d**2+(phi_d*smp.cos(theta))**2) +  smp.Rational(1,2)*B_d*(psi_d+phi_d*smp.sin(theta))**2

        K_r = smp.Rational(1,2)*A_r*(phi_d*smp.sin(theta))**2 + smp.Rational(1,2)*B_r*(chi_d-theta_d)**2

        v2_M =  (x_d - R*phi_d*smp.sin(theta)*smp.cos(phi))**2 + \
                (y_d - R*phi_d*smp.sin(theta)*smp.sin(phi))**2 + \
                (R*theta_d)**2 + \
                2*R*phi_d*smp.cos(theta)*(y_d*smp.cos(phi)-x_d*smp.sin(phi))

        v2_mu = (x_d - (R+l)*phi_d*smp.sin(theta)*smp.cos(phi))**2 + \
                (y_d - (R+l)*phi_d*smp.sin(theta)*smp.sin(phi))**2 + \
                ((R+l)*theta_d)**2 + \
                2*(R+l)*phi_d*smp.cos(theta)*(y_d*smp.cos(phi)-x_d*smp.sin(phi))

        V = M*g*R*smp.cos(theta) + mu*g*(R+l)*smp.cos(theta)
        
        L = K_d + K_r + smp.Rational(1,2)*M*v2_M + smp.Rational(1,2)*mu*v2_mu - V
        print("done defining L")  

        print("Defining E-L for theta")
        LE_theta = smp.diff(smp.diff(L, theta_d), t).simplify() - smp.diff(L, theta)
        print("Defining E-L for phi")
        LE_phi = smp.diff(smp.diff(L, phi_d), t).simplify() - smp.diff(L, phi)
        print("Defining E-L for chi")
        LE_chi = smp.diff(smp.diff(L, chi_d), t).simplify() - smp.diff(L, chi)
        print("Defining E-L for psi")
        LE_psi = smp.diff(smp.diff(L, psi_d), t).simplify() - smp.diff(L, psi)
        print("Done making E-L Equations!")

        print("Solving E-L Equations Simultaneously")
        sols = smp.solve([LE_theta, LE_phi, LE_chi, LE_psi], (theta_dd, phi_dd, chi_dd, psi_dd), minimal=True, simplify=False, Rational=False)
        print("done!")

        print("Simplifying E-L Equations")
        sols= smp.simplify(sols)
        print("done simplifying solutions")
        
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/solutions.txt', 'w') as f:
                f.write(str(sols))
        print("Solutions saved to file")

        print("Lambdifying Theta_dd")
        dz1dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[theta_dd])
        dthetadt = smp.lambdify(theta_d, theta_d)
        print("Lambdifying Phi_dd")
        dz2dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[phi_dd])
        dphidt = smp.lambdify(phi_d, phi_d)
        print("Lambdifying Chi_dd")
        dz3dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[chi_dd])
        dchidt = smp.lambdify(chi_d, chi_d)
        print("Lambdifying Psi_dd")
        dz4dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[psi_dd])
        dpsidt = smp.lambdify(psi_d, psi_d)
        print("Done Lambdifying Equations")
        print(dz1dt, dthetadt, dz2dt, dphidt, dz3dt, dchidt, dz4dt, dpsidt)

        dill.settings['recurse'] = True

        print('Saving Lambdas to file')
        lambdified_functions = {
        'dz1dt': dz1dt,
        'dthetadt': dthetadt,
        'dz2dt': dz2dt,
        'dphidt': dphidt,
        'dz3dt': dz3dt,
        'dchidt': dchidt,
        'dz4dt': dz4dt,
        'dpsidt': dpsidt
        }

        # Save lambdified functions to a file
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/lambdas.txt', 'wb') as f:
                dill.dump(lambdified_functions, f)
else:
        print('Loading lambdified functions from file')
        # Load lambdified functions from the file
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/lambdas.txt', 'rb') as f:
                loaded_lambdified_functions = dill.load(f)
        # Re-assign loaded lambdified functions
        dz1dt = loaded_lambdified_functions['dz1dt']
        dthetadt = loaded_lambdified_functions['dthetadt']
        dz2dt = loaded_lambdified_functions['dz2dt']
        dphidt = loaded_lambdified_functions['dphidt']
        dz3dt = loaded_lambdified_functions['dz3dt']
        dchidt = loaded_lambdified_functions['dchidt']
        dz4dt = loaded_lambdified_functions['dz4dt']
        dpsidt = loaded_lambdified_functions['dpsidt']
        print('Done Loaded lambdified functions from file')
        # Verify loaded lambdified functions
        print(dz1dt, dthetadt, dz2dt, dphidt, dz3dt, dchidt, dz4dt, dpsidt)


def dSdt(t, S, M, R, l, mu, g, A_d, B_d, A_r, B_r):
    theta, z1, phi, z2, chi, z3, psi, z4 = S
    return [
        dthetadt(z1), 
        dz1dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, z1, z2, z3, z4),
        dphidt(z2),
        dz2dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, z1, z2, z3, z4),
        dchidt(z3),
        dz3dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, z1, z2, z3, z4),
        dpsidt(z4),
        dz4dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, z1, z2, z3, z4) 
        ]
    
t = np.linspace(0, 10, 1001)
S0 = [theta_0, theta_d_0, phi_0, phi_d_0, chi_0, chi_d_0, psi_0, psi_d_0]
print("Solving ODE...")
#ans = odeint(dSdt, S0, t=t, rtol=0.01, args=(M_const, R_const, l_const, mu_const, g_const))
print("Solved!")

M = 1.0
R = 1.0
l = 1.0
mu = 1.0
g = 9.8
A_d = 1.0
B_d = 1.0
A_r = 1.0
B_r = 1.0

ans = solve_ivp(dSdt, (0.0, 10.0), S0, t_eval=t, args=(M, R, l, mu, g, A_d, B_d, A_r, B_r))
print("Solved!")
print(ans)

# Define the file path
file_path = '/Users/alex/Documents/VSC_Python/MECH_6681/output.csv'

# Save the 'ans' array to a CSV file
with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for t, the, th_d, ph, ph_d, ch, ch_d, ps, ps_d in zip(ans.t, ans.y[0], ans.y[1], ans.y[2], ans.y[3], ans.y[4], ans.y[5], ans.y[6], ans.y[7]):
                writer.writerow([t, the, th_d, ph, ph_d, ch, ch_d, ps, ps_d])
                
print("Output saved to CSV file:", file_path)




