''' This file is a lookup table for The OpenQASM instructions '''
''' Ref: https://github.com/qevedo/openqasm/tree/master/src/libs '''
''' Tutorial: https://quantum-computing.ibm.com/docs/iqx/operations-glossary '''


''' TODO:
    1. we should have a simple exclude/include mechanism for this colleciton so that one can easily
    mix and match various instructions that are available to the GA search space
'''

"""
the commands are organized as such:
   <string>: name of command
   followed by an array of ints representing: [<input> <arguments>]
   Examples: 
       U3(theta, phi, lambda) q -- 1 input (q), 3 arguments (theta, phi, lambda)
       x a -- 1 input (a), 0 argument    
"""
'''
Note that q should be in the form of q[0], q[1], etc. 
'''
COMMANDS = {
    #TODO: think about the 'c' control gate, and 'if' operation (if (c==0) x q[0];)

    ###############################################################################
    # Obsolete gates
    #u3 is obsolete and is now the u gate
    #"u3":   [1, 3],    #U(theta,phi,lambda) q; - u3(theta, phi, lam) q[0]; // theta, phi, lam = arguments, default is pi/2, q[0] = input
    #"u2":   [1, 2],    #U(phi/2, lambda) q; - u2(theta, phi) q[0]; // theta phi = arguments, defualt is pi/2, q[0] = input
    #u1 is obsolete and is now the phase gate
    #"u1":   [1, 1],    #U(0,0,lambda) q; - u1 (theta) q[0]; // theta = argument, default is pi/2, q[0] = input
    # "id":   [1, 0],    #idle gate // OBSOLETE!
    #"u0":   [1, 1],    #idle gate with length gamma *sqglen // ERROR!
    #"cz":   [2, 0],    #controlled-Phase - cz q[0], q[1];
    #"cy":   [2, 0],    #controlled-Y - cy q[0], q[1];
    #"ch":   [2, 0],    #controlled H; ch q[0], q[1];    
    #"cswap":[3, 0],    #cswap - cswap q[0], q[1], q[2];
    #"crx":  [2, 1],    #controlled rx rotation - crx(angle) q[0], q[1]; // angle is argument, default: pi/2
    #"cry":  [2, 1],    #controlled ry rotation - same as above
    #"crz":  [2, 1],    #controlled rz rotation - same as above
    #"cu1":  [2, 1],    #controlled phase rotation - cu1(angle) q[0], q[1]; // angle is arugment, default: pi/2
    #"cu3":  [2, 3],    #cu3(alpha1, alpha2, alpha3) q[0], q[1]; // same as above
    #"u0":   [1, 0],    #UNKNOWN (??)
    ###############################################################################

    #0
    "u":    [1, 3],     #u(theta, phi lam) q[0];  replaces u3, u2, cu3
    "p":    [1, 1],     #phase gate - this replaces u1, cu1
    "cx":   [2, 0],    #controlled NOT - cx q[0] q[1];
    "x":    [1, 0],     #Pauli gate: bit-flip: x q[0]
    "y":    [1, 0],     #Pauli gate: bit and phase flip - y q[0]; 
    #6
    "z":    [1, 0],     #Pauli gate: phase flip - z q[0];
    "h":    [1, 0],     #Clifford gate: Hadamard - h q[0];
    "s":    [1, 0],     #Clifford gate: sqrt(Z) phase gate - s q[0];
    "sdg":  [1, 0],   #Clifford gate: conjugate of sqrt(Z) - sdg q[0];
    "t":    [1, 0],     #C3 gate: sqrt(S) phase gate - t q[0];
    #11
    "tdg":  [1, 0],   #C3 gate: conjugate of sqrt(S) - tdg q[0];
    #standard rotations
    "rx":   [1, 1],    #Rotation around X-axis - rx(angle) q[0]; angle is argument, default: pi/2, q[0] is input 
    "ry":   [1, 1],    #rotation around Y-axis - ry(angle) q[0]; same as above
    "rz":   [1, 1],    #rotation around Z-axis - rz(angle) q[0]; same as above
    #QE Standard User-Defined Gates
    "swap": [2, 0],    #swap - swap q[0], q[1]
    #16
    "ccx":  [3, 0],    #toffoli ccx q[0], q[1], q[2];
    "rxx":  [2, 1],    #molmer-sorensen gate - rxx(angle) q[0], q[1];
    "rzz":  [2, 1],    #two-qubit ZZ rotation - rzz(angle) q[0], q[1];
    "sx":   [1, 0],    #square root not gate - sx q[0];
    "sxdg": [1, 0],    #square root not dagger gate - sxdg q[0];

    ##### unimplemented
    # barrier q;
    # reset q[0];
    # if (c==0) x q[0]; // c is a classic register
    # measure q[0];
}
"""
OBSOLETE:
COMMANDSARRAY = [
    "u3","u2","u1","cx",
    #"id",
    "u0","x","y","z","h","s","sdg","t","tgd","rx",
    "ry","rz","cz","cy","swap","ch","ccx","cswap","crz","cu1","cu3","rzz",
]
"""

