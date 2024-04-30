--------------------READ ME------------------------
##Project Overview

This project is the simulation of a unicycle represented with a disk and rotor using Lagrangian Mechanics, and a design of a suitable control system using feedback linearization for the graduate course MECH 6681 at Concordia University, Montreal. This is the first time the course has been offered, and provides students with the knowledge to model and control complex dynamical systems. 

The system this project focuses on is described in further detail in the following references:

[1] A.M. Bloch. Nonholonomic Mechanics and Control. 2nd ed. New York, USA: Springer-Verlag, 2015.
[2] D.V. Zenkov, A.M. Bloch, and J.E. Marsden. “Flat Nonholonomic Matching”. In: Proceedings of the 2002 American Control Conference (IEEE Cat. No.CH37301). Vol. 4. 2002, 2812–2817 vol.4. doi: 10.1109/ACC.2002.1025215.10

##Code Description
The "tri_sci_py.py" and "tri_sci_ctrl.py" files are scripts that use sympy, a symbolic library in python, to create and solve the Euler-Lagrange equations for the system and save them to files. Then, using the input-output feedback linearization taught in class, a control input in designed and added to the system, which is then solved for using a differential equation solver from "SciPy" and displayed using "MatPlotLib"

The non ".txt" and ".pkl" files are a "checkpoint" due to run time limitations of te code. 
It can take several minutes for "tri_sci_py.py" and "tri_sci_ctrl.py" to generate and solve the Euler-Lagrange equations, so for the control system design portion, simply set "READ_MODE" to "True", and it will skip the calculation of these and simply pull the differential equations from the ".pkl" files. The ".txt" files are simply for the option of carrying the equations over to MATLAB
