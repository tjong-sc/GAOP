from typing import List

''' 
This converts Primitive or OO Representation type into an executable OpenQASM program and then return the outputs
for the caller.

A sample program string is provided below:
"OPENQASM 2.0;        <--- programHeader
include "qelib1.inc"; <--- libraryHeader
qreg q[2];             <-- collection starts here
creg c[2];
h q[0];
cx q[0],q[1];          <-- colleciton ends here
measure q -> c; <--- ignored since we don't care about collapsing the quantum state on these registers.

Input args: string - dump from a collection of PrimitiveRepresentations
Outputs: probability - this can be computed for fitness or measurement if you want.
'''

class OpenQASMConverter:
    programHeader = "OPENQASM 2.0;\n"
    libraryHeader = "include \"qelib1.inc\";\n"
    registerPattern = "qreg q[{0}];\n"

    '''
    convertPrimitivesToOpenQASM - converts a collection of PrimitiveRepresentations into OpenQASM code for execution 
    '''
    @staticmethod
    def convertPrimitivesToOpenQASM(numOfRegisters:int, collection:List[str], prepareStatement:str = None) -> str:
        # supposedly this is the most efficient method to create a long string:
        # https://waymoot.org/home/python_string/

        return OpenQASMConverter.programHeader + OpenQASMConverter.libraryHeader + \
            OpenQASMConverter.registerPattern.format(numOfRegisters) + \
                ("" if prepareStatement == None else prepareStatement) +\
            "\n".join(primitive.toString() for primitive in collection)