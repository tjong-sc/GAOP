import sys
import math
import numpy as np
from GAOP.Representations.BasicInstruction import BasicInstruction
from GAOP.QiskitDrivers.QASMLookup import COMMANDS
from GAOP.Statistics.RandomWrapper import RandomWrapper
from typing import List

class QASMPrimitiveInstruction(BasicInstruction):
    """
    This class is used as the most basic GA gene representation where we are using offsets + values to represent QASM instructions
    """

    ''' 
    Parameters are determined based on the type of instructions since different instructions
    take on different number of parameters
    '''    
    defaultArgumentSeperator = ","    
    defaultCommandSeperator = " "    
    stringPatternWithNoArgument = "{0} {1};"
    stringPatternWithArgument = "{0}({1}) {2};"
    commandSuffix = ";\n"

    def __init__(self, rand: RandomWrapper, numOfRegisters:int = 1, instructionOffset:int = -1, arcFunc:callable = None):
        """ This sets up a basic instruction - randomized    
        The number of quantum registers will also be referred to as q[0], q[1], so on and so forth
        instructionOffset == -1 indicates to the init function to randomly generate an instruction
        """
        self.rand = rand
        self.numOfRegisters = numOfRegisters
        self.registers = [] #this records the registers for a particular instruction
        self.arguments = [] #this records the arguments pertaining to a register within an instruction
        if arcFunc == None:
            self.arcFunc = self.defaultArgumentFunc
        else:
            self.arcFunc = arcFunc
        self.createCommand(instructionOffset, self.arcFunc)

    def getClone(self) -> list:
        """ This performs deep copy of the instruction """
        #TODO: to recreate a random instructions during a clone is inefficient at best.... rewrite please
        clone = QASMPrimitiveInstruction(self.rand, numOfRegisters = self.numOfRegisters, arcFunc = self.arcFunc)
        clone.registers = self.registers.copy()
        clone.arguments = self.arguments.copy()
        clone.command = self.command
        clone.commandIndex = self.commandIndex
        return clone

    def getCompatibleCommands(self) -> list:
        """
        This method returns a list of compatible commands corresponding to the number of required registers.
        For instance, if we have only 2 registers available, then instructions requiring <= 2 registers are compatible 
        (we can't do cx q[0] q[0], so we need to have more than or equal amount of registers for an instruction to be valid)
        TODO: We should privatize this since it needs some variable initialization
        """
        resultarray = []
        #gets the number of inputs and arguments for this command and check against self.numOfRegisters
        for i in COMMANDS.keys():
            if((COMMANDS[i])[0] <= self.numOfRegisters):
                resultarray.append(i)
        return resultarray
            
    def createCommand(self, instructionOffset: int, argFunc: callable) -> None:
        """ 
        this method is used to randomize the parameters of a command
        TODO: We should privatize this since it needs some variable initialization 
        """
        if(instructionOffset >= 0):
            instOffset = instructionOffset
        else:
            instOffset = self.rand.randint(0, len(COMMANDS))
            
        #assigns the command
        self.command = list(COMMANDS.keys())[instOffset]
        self.commandIndex = instOffset 

        #gets the number of inputs and arguments for this command
        tuple = COMMANDS[self.command]
        numofRegs = tuple[0]
        numofArguments = tuple[1]
        
        #this is more like an ASSERT!
        self.invalidCommandCheck(numofRegs)

        #now let's do it for the arguments, default value is PI/2            
        if(numofArguments > 0):
            self.arguments = np.empty(numofArguments)
            self.arguments = list(map(argFunc, self.arguments))
        
        #fill up the list of registers with non-repeating numbers (which references the registers)
        #random.sample(range(100),10) - this returns 10 numbers from a range of [0,99].
        self.registers = self.rand.sample(range(self.numOfRegisters), numofRegs)
        
    def getCommand(self) -> None:
        pass
        
    def getInputs(self) -> None:
        """
        The getParameters method returns the input parameters (0 or more in the form of an array) 
        for this instruction
        """
        pass
           
    def populateRegisters(self, old_set:List[int], fromIndex:int) -> None:        
        """
        Note that we do not perform any sort of error checking on this, so only call this method
        when one's sure that we have sufficient registers for the self.registers array.

        This method populates the registers from a given start index until the end of the list. 
        TODO: we should combine this with randomizeRegister perhaps.
        """
        length = len(self.registers)
        #creates a list of incrementing numbers
        candidates = list(range(0, length))
        #do set deduction
        difference = [item for item in candidates if item not in old_set]
        #resample from the new set that we have obtained from set deduction
        results = self.rand.sample(difference, length - fromIndex)
        #sets up the new registers
        for i in range(0, len(results)):        
            self.registers[fromIndex] = results[i]
            fromIndex += 1
         
    def randomizeCommand(self, commandfunc:callable = None, argumentFunc:callable = None, doTruncation:bool = True) -> None:
        """
        Randomize command - this function preserves the arguments if possible or it will 
        expands the argument based on the argumentfunc if provided - while changing the commands.
        default value = true
        commandfunc is a parameterless function that returns a desirable command offset
        argumentFunc is a single parameter function that returns a desirable *mutation* of the arg.
        """
        # randomly generate a new command and truncate the arguments/registers as necessary
        if(commandfunc == None):
            compatCommands = self.getCompatibleCommands()
            instOffset =self.rand.sample(range(len(compatCommands)), 1)[0]
        else:
            instOffset = commandfunc()
        
        if(argumentFunc == None):
            argumentFunc = self.defaultArgumentFunc

        #assigns the new command
        self.command = list(COMMANDS.keys())[instOffset]
        self.commandIndex = instOffset 

        #gets the number of inputs and arguments for this command
        tuple = COMMANDS[self.command]
        numofRegs = tuple[0]
        numofArguments = tuple[1]
        
        #this is more like an ASSERT!
        self.invalidCommandCheck(numofRegs)
        
        #proceed to truncate or expand the args if necessary
        oldArglen = len(self.arguments)
        if(numofArguments > 0):
            if(doTruncation): #preserves the value of the existing elements
                self.arguments = np.resize(self.arguments, numofArguments)
                #we should apply the argfunc on the new arguments if any
                if(len(self.arguments) - oldArglen > 0):
                    for i in range(oldArglen-1, numofArguments):
                        self.randomizeArgument(i, argumentFunc)
            else:
                self.arguments = np.resize(numofArguments)
                self.arguments.fill(argumentFunc)                
        else:
            self.arguments = []
        
        oldReglen = len(self.registers)
        if(doTruncation): #preserves the value of the existing elements
            old_registers = self.registers.copy()
            self.registers = np.resize(self.registers, numofRegs)
            #we will populate the new array elements with existing registers, if applicable.
            if(len(self.registers) - oldReglen > 0):
                self.populateRegisters(old_registers, oldReglen)
        else: #otherwise we will just resize the array and populate them from a random sample (no replacement)
            self.registers = np.resize(numofRegs)
            self.registers = self.rand.sample(range(self.numOfRegisters), numofRegs)
        
    def randomizeRegister(self, offset:int) -> None:
        """
        This method randomizes the referenced register for a particular offset.
        idea: 1. create a set from the existing register, removes the current register from the set since we may
          potentially reuse it after randomization. 
          2. create a new sampling set from the unused register + the current regsiter
          3. fill in self.registers[offset] with 1 sample from the sampling
        """
        # all registers are used up so there's nothing to randomize for a particular offset.
        if(len(self.registers) == self.numOfRegisters):
            return
        #generates a list of range of numbers
        a_set = set(range(0,self.numOfRegisters))
        regs = set(self.registers)
        #do set deduction on other used registers (not ref by offset) so that we have
        #the remaining unused registers + whatever that's ref by offset
        difference = [reg for reg in a_set if reg not in regs or reg == self.registers[offset]]

        self.registers[offset] = difference[self.rand.randint(0, len(difference))]
        
    def randomizeArgument(self, offset:int, myfunc:callable, suppressException:bool = False) -> None:
        """
        The randomize method randomizes the inputs based on a given seeded random number generator
        Inputs: 
        offset: argument offset
        func: a 1 parameter input arg lambda function. This parameter may exert some influence on the final output value.
        """
        if(myfunc == None):
            myfunc = self.defaultArgumentFunc
        # we will randomize the arguments with the given func as mapping or just a random mutation
        if(len(self.arguments) > offset):
            self.arguments[offset] = myfunc(self.arguments[offset])
        else:
            raise Exception("Attempting to randomize an argument beyond the bounds of the array")
    
    def invalidCommandCheck(self, numOfRegisters:int) -> None: 
        #this is a check that prevents an operation that has no effect on the register is being created.
        #The program should terminate since this is a big error.
        if(numOfRegisters <= 0):            
            raise Exception(self.command + "is not a valid command since it doesn't operate on a register.")

        #now we need to check that the number of registers is ok for a command. Here are the scenarios:
        #1: the provided number of registers (stored in self.numOfRegisters) is >= what's required by the command: OK            
        #2: the provided number of registers is < what's required by the command: not ok, so we will reloop
        if(self.numOfRegisters < numOfRegisters ):
            if(self.__debug):
                print("Instruction {0} requires {1} registers and that's more than what's available: {2}".format(self.command, numOfRegisters, self.numOfRegisters))
            raise Exception(self.command + " is not a valid command for the given sent of inputs")
        
    def toString(self) -> str:     
        """
        The toString method creates a string out of this instruction. This method can be used to print human
        readable strings or send to the QiskitAdapter for execution
        """           
        localarray = ["q[" + str(reg) for reg in self.registers]
        localarray = [reg + "]" for reg in localarray]

        #the way that an instruction is formatted is based on the following observations:
        # no argument: ccx q[0] q[1]... etc
        # arguments: ccx(arg1, arg2, ...) q[0]...etc        
        if(len(self.arguments) > 0):            
            return self.stringPatternWithArgument.format(self.command, self.defaultArgumentSeperator.join(str(x) for x in self.arguments),
                                                         self.defaultArgumentSeperator.join(localarray))
        else:
            return self.stringPatternWithNoArgument.format(self.command, self.defaultArgumentSeperator.join(localarray))

    def defaultArgumentFunc(self, val):
        return round(self.rand.uniform(0, math.pi*2), 4)