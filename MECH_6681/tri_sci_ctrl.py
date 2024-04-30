import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import sympy as smp
# from matplotlib import animation
# from matplotlib.animation import PillowWriter
import csv
import pickle
import dill
from sympy.printing.mathml import mathml

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
theta, phi, chi, psi, x, y, u = smp.symbols('theta phi chi psi x y u', cls=smp.Function)
#Define the symbols for constants
A_d, B_d, A_r, B_r, M, R, l, mu, g, t = smp.symbols('A_d B_d A_r B_r M R l mu g t')

#Create Functions for the variables
theta = theta(t)
phi = phi(t)
chi = chi(t)
psi = psi(t)
x = x(t)
y = y(t)
u = u(t)

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
        #-------------------------------------Define Variables in the Simplified Lagrange Equation:-------------------------------------- 
        #------------------ ( L = Kd + Kr + 0.5M*v_M^2 + 0.5mu*v_mu^2 - V ) -------------------
        K_d = smp.Rational(1,2)*A_d*(theta_d**2+(phi_d*smp.cos(theta))**2) +  smp.Rational(1,2)*B_d*(psi_d+phi_d*smp.sin(theta))**2

        K_r = smp.Rational(1,2)*A_r*(phi_d*smp.sin(theta))**2 + smp.Rational(1,2)*B_r*(chi_d-theta_d)**2

        v2_M =  (x_d - R*phi_d*smp.sin(theta)*smp.cos(phi))**2 + \
                (y_d - R*phi_d*smp.sin(theta)*smp.sin(phi))**2 + \
                (R*theta_d)**2 #+ \
                #2*R*phi_d*smp.cos(theta)*(y_d*smp.cos(phi)-x_d*smp.sin(phi))

        v2_mu = (x_d - (R+l)*phi_d*smp.sin(theta)*smp.cos(phi))**2 + \
                (y_d - (R+l)*phi_d*smp.sin(theta)*smp.sin(phi))**2 + \
                ((R+l)*theta_d)**2 #+ \
                #2*(R+l)*phi_d*smp.cos(theta)*(y_d*smp.cos(phi)-x_d*smp.sin(phi))

        V = M*g*R*smp.cos(theta) + mu*g*(R+l)*smp.cos(theta)
        
        L = K_d + K_r + smp.Rational(1,2)*M*v2_M + smp.Rational(1,2)*mu*v2_mu - V
        print("done defining L")  
        
        #-----------------------------Define and Solve the Euler-Lagrange Equations for each Variable:----------------------
        #define
        print("Defining E-L for theta")
        LE_theta = smp.diff(smp.diff(L, theta_d), t).simplify() - smp.diff(L, theta)
        print("Defining E-L for phi")
        LE_phi = smp.diff(smp.diff(L, phi_d), t).simplify() - smp.diff(L, phi)
        print("Defining E-L for chi")
        LE_chi = smp.diff(smp.diff(L, chi_d), t).simplify() - smp.diff(L, chi) - u
        print("Defining E-L for psi")
        LE_psi = smp.diff(smp.diff(L, psi_d), t).simplify() - smp.diff(L, psi) - u
        print("Done making E-L Equations!")
        
        #solve
        print("Solving E-L Equations Simultaneously")
        sols = smp.solve([LE_theta, LE_phi, LE_chi, LE_psi], (theta_dd, phi_dd, chi_dd, psi_dd), minimal=True, simplify=False, Rational=False)
        print("Simplifying E-L Solutions")
        sols= smp.simplify(sols)
        print("done simplifying solutions")
        
        #lambdify
        
        q1 = smp.lambdify(theta, theta)
        q2 = smp.lambdify(theta_d, theta_d)
        q3 = smp.lambdify(phi, phi)
        q4 = smp.lambdify(phi_d, phi_d)
        q5 = smp.lambdify(chi, chi)
        q6 = smp.lambdify(chi_d, chi_d)
        q7 = smp.lambdify(psi, psi)
        q8 = smp.lambdify(psi_d, psi_d)
        
        
        print("Lambdifying Theta_dd")
        dq1dt = q2
        dq2dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[theta_dd])
        print("Lambdifying Phi_dd")
        dq3dt = q4
        dq4dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[phi_dd])
        print("Lambdifying Chi_dd")
        dq5dt = q6
        dq6dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[chi_dd])
        print("Lambdifying Psi_dd")
        dq7dt = q8
        dq8dt = smp.lambdify((t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, theta_d, phi_d, chi_d, psi_d), sols[psi_dd])
        print("Done Lambdifying Equations")
        
        #--------------------------------Write solution, latex, and lambdified functions to file---------------------------------
        # Save the solutions to a file
        dill.settings['recurse'] = True
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/solutions_ctrl.pkl', 'wb') as f:
                dill.dump(sols, f)
        print("Solutions saved to file")
                
        # Save the latex representation of the solutions to a file
        print("writing latex to file")
        smp.init_printing(use_unicode=True)
        print(smp.latex(sols))
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/latex.txt', 'w') as f:
                f.write(smp.latex(sols))
        print("Latex saved to file")
        
        # Save the lambdified functions to a file
        print('Saving Lambdas to file')
        lambdified_functions = {
        'q1': q1,
        'q2': q2,
        'q3': q3,
        'q4': q4,
        'q5': q5,
        'q6': q6,
        'q7': q7,
        'q8': q8,
        'dq1dt': dq1dt,
        'dq2dt': dq2dt,
        'dq3dt': dq3dt,
        'dq4dt': dq4dt,
        'dq5dt': dq5dt,
        'dq6dt': dq6dt,
        'dq7dt': dq7dt,
        'dq8dt': dq8dt
        }
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/lambdas ctrl.txt', 'wb') as f:
                dill.dump(lambdified_functions, f)
else:
        # Load the saved solutions from file
        print('Loading solutions from file')
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/solutions_ctrl.pkl', 'rb') as f:
                sols = dill.load(f)
        
        print('Loading lambdified functions from file')
        # Load lambdified functions from the file
        with open('/Users/alex/Documents/VSC_Python/MECH_6681/lambdas ctrl.txt', 'rb') as f:
                loaded_lambdified_functions = dill.load(f)
        # Re-assign loaded lambdified functions
        q1 = loaded_lambdified_functions['q1']
        q2 = loaded_lambdified_functions['q2']
        q3 = loaded_lambdified_functions['q3']
        q4 = loaded_lambdified_functions['q4']
        q5 = loaded_lambdified_functions['q5']
        q6 = loaded_lambdified_functions['q6']
        q7 = loaded_lambdified_functions['q7']
        q8 = loaded_lambdified_functions['q8']
        dq1dt = loaded_lambdified_functions['dq1dt']
        dq2dt = loaded_lambdified_functions['dq2dt']
        dq3dt = loaded_lambdified_functions['dq3dt']
        dq4dt = loaded_lambdified_functions['dq4dt']
        dq5dt = loaded_lambdified_functions['dq5dt']
        dq6dt = loaded_lambdified_functions['dq6dt']
        dq7dt = loaded_lambdified_functions['dq7dt']
        dq8dt = loaded_lambdified_functions['dq8dt']
        print('Done Loading lambdified functions from file')        
        

q = [q1, q2, q3, q4, q5, q6, q7, q8]
dqdt = [
        dq1dt(q2), 
        dq2dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8),
        dq3dt(q4),
        dq4dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8),
        dq5dt(q6),
        dq6dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8),
        dq7dt(q8),
        dq8dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8) 
        ]




"""def dSdt(t, S, M, R, l, mu, g, A_d, B_d, A_r, B_r):
    theta, q2, phi, q4, chi, q6, psi, q8 = S
    return [
        dq1dt(q2), 
        dq2dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8),
        dq3dt(q4),
        dq4dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8),
        dq5dt(q6),
        dq6dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8),
        dq7dt(q8),
        dq8dt(t, M, R, l, mu, g, A_d, B_d, A_r, B_r, theta, phi, chi, psi, q2, q4, q6, q8) 
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
                
print("Output saved to CSV file:", file_path)"""




