import typing
from GAOP.Statistics.RandomWrapper import RandomWrapper
from GAOP.Representations.BasicInstruction import Symbols

class Chromosome:
    """
    The chromosome class - a random instruction set is generated when an instance of this class is created.
    """
    def __init__(self, rand: RandomWrapper, sizeOfRegisters:int, generationFunction :callable, size:int , arcFunc:callable = None ):
        """
        The constructor expects a generationFunction, a default is provided in QASMPrimitiveInstruction
        fitnesses = the probability computed for each qubit state.
        """
        self.rand = rand
        self.size = size
        self.arcFunc = arcFunc
        self.numOfRegisters = sizeOfRegisters
        self.generationFunc = generationFunction
        self.probabilities = ()
        self.instructions = []
        self.fitness = -1
        for i in range(0, size):
            self.instructions.append(generationFunction(self.rand, numOfRegisters = sizeOfRegisters, arcFunc = arcFunc)) 
    
    def getClone(self) -> 'Chromosome':
        """ This method returns a clone (deepcopied) of this individual """
        #TODO: to recreate a random chromosome during a clone is inefficient at best.... rewrite please
        clone = Chromosome(self.rand, self.numOfRegisters, self.generationFunc, self.size, self.arcFunc)
        clone.setProbabilities(self.probabilities.copy())
        clone.instructions = []
        clone.fitness = self.fitness
        for instruction in self.instructions:
            clone.instructions.append(instruction.getClone())
        return clone

    def setProbabilities(self, probabilities: dict) -> None:
        self.probabilities = probabilities

    def getProbabilities(self) -> dict:
        return self.probabilities

    def mutateInstruction(self, instructionOffset:int, commandfunc:callable = None, argumentFunc:callable = None, doTruncation:bool = True) -> None:
        """        
        This method mutates the instruction only 
        """
        self.instructions[instructionOffset].randomizeCommand(commandfunc, argumentFunc, doTruncation)
    
    def mutateArgument(self, instructionOffset:int, argumentOffset:int, argumentFunc:callable = None, suppressException:bool = False) -> None:
        """
        This method mutates the argument
        """
        self.instructions[instructionOffset].randomizeArgument(argumentOffset, argumentFunc, suppressException)
        
    def mutateRegisters(self, instructionOffset:int, registerOffset:int) -> None:
        """
        This method mutates the registers
        """
        self.instructions[instructionOffset].randomizeRegister(registerOffset)

    def uniformMutation(self, mutationProbability:float, commandfunc:callable = None, \
        argumentFunc:callable = None, doTruncation:bool = True) -> None:
        """
        #TODO: I need to debug this properly
        This is a mutation operator that randomly mutates either the instruction, argument or register, 
        treating them the same.
        """
        val = self.rand.uniform(0.0, 1.0)
        #a mutation should happen when the random number is <= the given mutationProbability
        if(val <= mutationProbability):
            #roll a random number to determine if it's instruction/register/argument
            mutType = self.rand.randint(0, len(Symbols))
            #roll another random number to determine the instruction within an individual that should be mutated
            instructionOffset = self.rand.randint(0, len(self.instructions))
            #performs the corresponding mutation
            if(mutType == Symbols.INSTRUCTION.value):
                print("\t-->Mutation:Instruction\n\tBefore:{0}\n".format(self.instructions[instructionOffset].toString()))
                self.instructions[instructionOffset].randomizeCommand(commandfunc, argumentFunc, doTruncation)
                print("\t<--To:{0}".format(self.instructions[instructionOffset].toString()))
            elif (mutType == Symbols.REGISTER.value):
                print("\t-->Mutation:Register\n\tBefore:{0}\n".format(self.instructions[instructionOffset].toString()))
                instNumOfRegs = len(self.instructions[instructionOffset].registers)
                regOffset = self.rand.randint(0, instNumOfRegs)
                self.instructions[instructionOffset].randomizeRegister(regOffset)
                print("\t<--To:{0}\n".format(self.instructions[instructionOffset].toString()))
            else: #Argument
                print("\t-->Mutation:Argument\n\tBefore:{0}\n".format(self.instructions[instructionOffset].toString()))
                argLen = len(self.instructions[instructionOffset].arguments) 
                if(argLen > 0):
                    argOffset = self.rand.randint(0, argLen)
                    self.instructions[instructionOffset].randomizeArgument(argOffset, argumentFunc)
                print("\t<--To:{0}\n".format(self.instructions[instructionOffset].toString()))
  
    def instructionsMultipointCrossover(self, mate: 'Chromosome', off1: int = -1, off2: int = -1) -> 'Chromosome':
        """  
        This is a multipoint crossover operator.
        TODO: we should be able to combine instructionsMultipointCrossover with instructionsSinglePointCrossover
        Inputs: ind1Offset1 = instruction index of the first individual
            ind1Offset2 (inclusive!)= 2nd instruction index of the first individual 
            ind1Index = index of the first individual
            ind2Offset1 = instruction index of the first individual
            ... etc
        """
        #only work on clones
        parent1 = self.getClone()
        parent2 = mate.getClone()

        #generates a set of random numbers that are sorted to prevent offset1 > offset2
        set1 = [self.rand.randint(0, len(parent1.instructions)) for _ in range(2)]
        set2 = [self.rand.randint(0, len(parent2.instructions)) for _ in range(2)]
        set1.sort()
        set2.sort()
        ind1Offset1 = set1[0]
        ind1Offset2 = set1[1]
        ind2Offset1 = set2[0]
        ind2Offset2 = set2[1]
        print("\t---- Crossover, Offsets: P1: {0}, {1} P2: {2}, {3}".format(ind1Offset1, ind1Offset2, ind2Offset1, ind2Offset2))
        print("\t{0}\n\t{1}\n".format(parent1.toString(""), parent2.toString("")))
        #exchanges the instructions between the individuals starting from the index
        temparr1 = parent1.instructions[ind1Offset1:ind1Offset2]
        temparr2 = parent2.instructions[ind2Offset1:ind2Offset2]

        parent1.instructions = parent1.instructions[0:ind1Offset1] + \
            temparr2 + parent1.instructions[ind1Offset2:]
        parent2.instructions = parent2.instructions[0:ind2Offset1] + \
            temparr1 + parent2.instructions[ind2Offset2:]
        print("\tResults: \n\t{0}\n\t{1}\n\t-----End-----".format(parent1.toString(""), parent2.toString("")))
        return parent1, parent2

    def instructionsSinglePointCrossover(self, mate:'Chromosome') -> None:
        """
        TODO1: the individuals may have different numOfRegisters in the future. we need a correction mechanism in place
        TODO2: we might wanna support indifferent crossover in the future, so a correction mechanism needs to be there.
        """
        #only work on clones
        parent1 = self.getClone()
        parent2 = mate.getClone()
        #generates a set of random numbers that are sorted to prevent offset1 > offset2
        idx1 = self.rand.randint(0, len(parent1.instructions))
        idx2 = self.rand.randint(0, len(parent2.instructions))
        #exchanges the instructions between the individuals starting from the index
        temparr = parent2.instructions[idx2:]
        parent2.instructions[idx2:] = parent1.instructions[idx1:]
        parent1.instructions[idx1:] = temparr


    def operandsFlattenSinglePointCorssover(self) -> None:
        """
        TODO: The operand crossover combines all of the operands that show up in a program and perform crossover on them
        """
        pass 
    def operandsFlattenMultiPointCorssover(self) -> None:
        pass 

    def toString(self, prefix :str) -> str:
        output = prefix.join(primitive.toString() for primitive in self.instructions)
        return output
