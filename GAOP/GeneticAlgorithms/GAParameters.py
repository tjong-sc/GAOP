class GAParameters:
    """ This is a class that manages the parameters of GA """
    def __init__(self, probM:float, probC:float, totalEpochs:int, decreaseM:bool = False, \
        decreaseC:bool = False, MStartEpoch:int = 0, MEndEpoch:int = 0, \
        CStartEpoch:int = 0, CEndEpoch:int = 0, eliteRatio:float = 0.0, \
        openQASMPrepareStatement:str = None):
        """
        The decreaseM parameter is used to control the gradual decrease in mutation
        and vice verse for decreaseC
        MEpochStart is the point where we start reducing the mutation ratio until it hits
        0 at MEpochEnd (a straight curve line)
        The same applies to CEpochStart and CEpochEnd
        """
        self.probM = probM
        self.probC = probC
        self.decreaseM = decreaseM
        self.decreaseC = decreaseC
        self.MStartEpoch = MStartEpoch
        self.CStartEpoch = CStartEpoch
        self.totalEpochs = totalEpochs
        self.CEndEpoch = CEndEpoch
        self.MEndEpoch = MEndEpoch
        if(CEndEpoch == 0):
            self.Cslope = 0.0
        else:
            self.Cslope = -probC / (CEndEpoch - CStartEpoch)
        if(MEndEpoch == 0):
            self.MSlope = 0.0
        else:
            self.Mslope = -probM / (MEndEpoch - MStartEpoch)
        self.eliteRatio = eliteRatio
        self.openQASMPrepareStatement = openQASMPrepareStatement

    def getMRate(self, currentEpoch:int):
        """
        We need to handle 3 cases, namely:
        1) currentEpoch <= MStartEpoch
        2) MStartEpoch <= currentEpoch <= MEndEpoch
        3) currentEpoch <= MEndEpoch
        """
        #case 1
        if(currentEpoch <= self.MStartEpoch):
            return self.probM
        #case 2
        elif(currentEpoch > self.MEndEpoch): 
            return 0.0
        else: #case 3
            return self.probM + self.Mslope * (currentEpoch - self.MStartEpoch)

    def getCRate(self, currentEpoch:int):
        """
        We need to handle 3 cases, namely:
        1) currentEpoch <= CStartEpoch
        2) CStartEpoch <= currentEpoch <= CEndEpoch
        3) currentEpoch <= CEndEpoch
        """
        #case 1
        if(currentEpoch <= self.CStartEpoch):
            return self.probC
        #case 2
        elif(currentEpoch > self.CEndEpoch): 
            return 0.0
        else: #case 3
            return self.probC + self.Cslope * (currentEpoch - self.CStartEpoch)
