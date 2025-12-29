import pandas as pd
import matplotlib.pyplot as plt

totalTime = 31084032
df = pd.read_excel('ThroughputCalculations.xlsx')

totalTime = int(df["Time to output (CLK CYCLES)"][30])

timeToOutput = []
inputChannelsCompleted = []
inputChannelsUsedDWCV = []
inputChannelsUsedPWCV = [] # For pointwise each output channel needs to have a seperate input channels used array
totalInputChannels = []
totalOutputChannels = []
spatialSize = []
layerType = []
stride = []
pwcvCreated32 = []
dwcvCreated32 = []
pwcvFinishedWork = []
itemsToRemove = [] # A list of all "dead" weights which no longer need to be used. That way, when the memory needs to add an element, it knows
                   # what element to take out. Weights are stored as tuples: (layer, output channel, input channel block). Depthwise weights
                   # only use the first two arguments. Pointwise weights use the third as well.

for idx, row in df.iterrows():
    if(idx >= 1 and idx < 27):
        inputChannelsCompleted.append(0)
        inputChannelsUsedDWCV.append(0)        
        totalInputChannels.append(int(row["Input Channels"]))
        totalOutputChannels.append(int(row["Output Channels"]))
        spatialSize.append(int(row["Spatial Size (assume square)"]))
        timeToOutput.append(int(row["Time to output (CLK CYCLES)"]))
        layerType.append(row["Layer Type"])
        stride.append(int(row["Stride"]))
        
        inputChannelsUsedPWCV.append([])
        
        for channel in range(totalOutputChannels[idx - 1]):
            inputChannelsUsedPWCV[idx - 1].append(0)
            
class memory:
    
    # Storage is measured in bytes
    # Throughput is measured in bytes per clock cycle
    def __init__(self, DDRstorage, BRAMstorage, DDRthroughput, BRAMthroughput):
        self.DDRStorage = DDRstorage #We low key may not need this, and we can just assume DDR has infinite memory
        self.BRAMStorage = BRAMstorage
        self.DDRthroughput = DDRthroughput
        self.BRAMthroughput = BRAMthroughput
        self.BRAMused = 0
        
        # Every index of the weight list is a layer. Every element (layer) points to another list. That list shows what weights are loaded in the weight list.
        # For depthwise convolution it's simple, each input channel corresponds to an output channel, so you just have a list where every index is an output
        # channel, 1 for kernel loaded, 0 for kernel not loaded.
        # For pointwise convolution it's less simple, each output channel has one kernel, but that kernel may be more than the pointwise convolution unit has
        # access to. So we split the kernel into groups of 32. So even though we have a kernel with [input channels] number of weights, the number of kernels
        # is [input channels/32]. So, every index corresponds to an output channel, which points to another list, showing which groups of 32 weights have been
        # loaded from the input channels.
        # So, just a reminder, here is the structure of the list:
        # weightList = [layer0, layer1, layer2, layer3, ..., layerN]
        #                  |         |
        #     (Depthwise) \|/        |
        # [9, 9, 9, 5, ..., 0, 0]    |
        # (Index signifies output    |
        # channel, the number        |
        # signifies how many         |
        # out of the 9 needed        |
        # for a kernel are present)  |
        #                           \|/ (Pointwise)
        #           [OutChnl0, OutChnl1, OutChnl2, ...]
        #               |
        #              \|/
        #      [32, 32, 15, ..., 0, 0]
        # (Index signifies group of 32
        # input channels. The number present
        # signifies the number of weights out
        # of that group of 32 which are currently
        # loaded.)
        
        self.weightList = []
        
        # Initialization
        for idx in range(28):
            self.weightList.append([])
            if(layerType[idx] == "Depthwise"):
                for outChannel in range(totalOutputChannels[idx]):
                    self.weightList[idx].append(0)
            elif(layerType[idx] == "Pointwise"):
                for outChannel in range(totalOutputChannels[idx]):
                    self.weightList[idx].append([])
                    for inChannel in range(totalInputChannels[idx]//32):
                        self.weightList[idx][outChannel].append(0)
            else:
                raise RuntimeError("Error: Found a layer which is neither pointwise or depthwise")
        
    # Input channel block only used for pointwise convolution
    def loadWeights(self, layer, outputChannel, inputChannelBlock):
        # We need an algorithm to determine what channel to take out and what channel to take in
        if(self.memoryFull()):
            raise RuntimeError("Error: Can't load weights if memory is currently full")
        
        
        self.BRAMused += 1
        
        if(layerType[layer] == "Depthwise"):
            self.weightList[layer][outputChannel] += 1
        elif(layerType[layer] == "Pointwise"):
            self.weightList[layer][outputChannel][inputChannelBlock] += 1
        else:
            raise RuntimeError("Error: Found a layer which is neither pointwise or depthwise")
        
    def contains(self, layer, outputChannel, inputChannelBlock):
        if(layerType[layer] == "Depthwise"):
            return self.weightList[layer][outputChannel]
        elif(layerType[layer] == "Pointwise"):
            return self.weightList[layer][outputChannel][inputChannelBlock]
        else:
            raise RuntimeError("Error: Found a layer which is neither pointwise or depthwise")
    
    def memoryFull(self):
        if(self.BRAMused == self.BRAMStorage and len(itemsToRemove) == 0):
            return True
        else:
            return False

mem = memory()

class DWCV:
    
    def __init__(self):
        self.dwcvCounter = 0
        self.dwcvActive = []
        self.layerWorkingOn = -1
    
    def updateActivity(self):
        if(self.dwcvCounter > 0):
            self.dwcvActive.append(1)
            self.dwcvCounter -= 1
            if(self.dwcvCounter == 0):
                inputChannelsCompleted[self.layerWorkingOn] += 1
                if(inputChannelsCompleted[self.layerWorkingOn] % 32 == 0):
                    dwcvCreated32.append(clkCycles)
                self.layerWorkingOn = -1
        else:
            self.dwcvActive.append(0)
    
    def startWorking(self, clockCycles, layer):
        if(mem.contains(layer - 1, inputChannelsUsedDWCV[layer - 1]) != 9):
            RuntimeError("Error: Depthwise Convolution was attempted to start without kernel loaded")
            
        self.dwcvCounter = clockCycles
        self.layerWorkingOn = layer
        inputChannelsUsedDWCV[layer - 1] += 1
        for idx in range(9):
            itemsToRemove.append((layer - 1, inputChannelsUsedDWCV[layer - 1], 0))
        
    def isActive(self):
        if(self.dwcvCounter > 0):
            return True
        else:
            return False
        
    def getActivity(self):
        return self.dwcvActive
        
class PWCV:
    
    def __init__(self, numPWCV):
        self.pwcvCounter = 0
        self.pwcvActive = []
        self.layerWorkingOn = -1
        self.numPWCV = numPWCV
        self.inputChannelsWorkingOn = []
    
    def updateActivity(self):
        if(self.pwcvCounter > 0):
            self.pwcvActive.append(1)
            self.pwcvCounter -= 1
            if(self.pwcvCounter == 0):
                pwcvFinishedWork.append(clkCycles)
                if(all(inputChannelsUsedPWCV[self.layerWorkingOn - 1][inputChannel] == totalInputChannels[self.layerWorkingOn - 1] for inputChannel in self.inputChannelsWorkingOn)):
                    inputChannelsCompleted[self.layerWorkingOn] += 32 # Since PWCV unit works on batches of 32 in parallel
                    pwcvCreated32.append(clkCycles)
                self.layerWorkingOn = -1
                self.inputChannelsWorkingOn = []
        else:
            self.pwcvActive.append(0)
    
    def startWorking(self, clockCycles, layer, inputChannels):
        for inputChannel in inputChannels:
            if(mem.contains(layer - 1, inputChannel, (inputChannelsUsedPWCV[layer - 1][inputChannel] + 32) // 32) != 32):
                RuntimeError("Error: Pointwise Convolution was attempted to start without kernel loaded")
        
        self.pwcvCounter = clockCycles//self.numPWCV
        self.layerWorkingOn = layer
        self.inputChannelsWorkingOn = inputChannels #inputChannels is a list of input channels the PWCV unit is working on concurrently
        for inputChannel in inputChannels:
            for idx in range(32):
                itemsToRemove.append((layer, inputChannel, (inputChannelsUsedPWCV[layer - 1][inputChannel] + 32) // 32))
            inputChannelsUsedPWCV[layer - 1][inputChannel] += 32
        
        
    def isActive(self):
        if(self.pwcvCounter > 0):
            return True
        else:
            return False
    
    def getActivity(self):
        return self.pwcvActive
    

clkCycles = 0
inputChannelsCompleted[0] = totalInputChannels[0] # Should be 32, assume all input channels have been finished by standard convolution
layerWorkingOn = 1
completedLayers = [] # Number of clock cycles at which layers got completed
completedLayers.append(0)

dwcvUnit = DWCV()
pwcvUnits = PWCV(2)

# totalTime = 1250000 * 2

for clkCycles in range(totalTime):
    if(not dwcvUnit.isActive()):
        for idx in range(1, len(inputChannelsCompleted)):
            if(inputChannelsCompleted[idx] < totalInputChannels[idx] and inputChannelsCompleted[idx - 1] > inputChannelsUsedDWCV[idx - 1]):
                if(layerType[idx - 1] == "Depthwise"):
                    dwcvUnit.startWorking((timeToOutput[idx - 1]/totalInputChannels[idx]), idx)
    
    if(not pwcvUnits.isActive()):
        for idx in range(1, len(inputChannelsCompleted)):
            if(inputChannelsCompleted[idx] < totalInputChannels[idx] and layerType[idx - 1] == "Pointwise"):
                for channelBlock in range(totalInputChannels[idx]//32):
                    condition = True
                    channelsWorkingOn = []
                    for channel in range(32):
                        if(inputChannelsUsedPWCV[idx - 1][channel + 32*channelBlock] <= inputChannelsCompleted[idx - 1] - 32):
                            condition = condition and True
                            channelsWorkingOn.append(channel + 32*channelBlock)
                        else:
                            condition = condition and False
                            break
                    if condition is True:
                        pwcvUnits.startWorking(timeToOutput[idx-1]/(totalInputChannels[idx]/32) * (1/(totalInputChannels[idx-1]/32)), idx, channelsWorkingOn)
                        break
                break #We can stop on the first condition this is true

    dwcvUnit.updateActivity()
    pwcvUnits.updateActivity()
    
    if(inputChannelsCompleted[layerWorkingOn] == totalInputChannels[layerWorkingOn]):
        completedLayers.append(clkCycles)
        layerWorkingOn += 1
        
    if(layerWorkingOn == len(inputChannelsCompleted)):
        break
        
    if clkCycles % 10000 == 0:
        print("Cycle: ", clkCycles)
        print("pwcv.isActive(): ", pwcvUnits.isActive())
        print("dwcv.isActive(): ", dwcvUnit.isActive())
        print("pwcv count: ", pwcvUnits.pwcvCounter)
        print("dwcv count: ", dwcvUnit.dwcvCounter)

dwcvActivity = dwcvUnit.getActivity()
pwcvActivity = pwcvUnits.getActivity()

downSample = 1000

plt.plot(dwcvActivity[::downSample], label="dwcv")
plt.plot(pwcvActivity[::downSample], label="pwcv")

plt.plot([completedLayer/downSample for completedLayer in completedLayers], range(len(completedLayers)), label="Completed Layers")
plt.scatter([layerCompletion/downSample for layerCompletion in pwcvCreated32], [0] * len(pwcvCreated32), marker="o", label="Times at which the PWCV Unit produced 32 channels", color="orange")
plt.scatter([layerCompletion/downSample for layerCompletion in dwcvCreated32], [0] * len(dwcvCreated32), marker="*", label="Times at which the DWCV Unit produced 32 channels", color="blue")

plt.legend()        # <-- this makes the labels appear

plt.show()

print("Completed layers: ")
for idx in range(len(completedLayers)):
    print("Layer " + str(idx) + " was completed at clk cycles " + str(completedLayers[idx]))

print("It took " + str(clkCycles) + " clock cycles to finish")

print("The number of clock cycles between the last two PWCV finishing 32 is: " + str(pwcvCreated32[-1] - pwcvCreated32[-2]))
print("The number of clock cycles between the last two PWCV weight switches is: " + str(pwcvFinishedWork[-1] - pwcvFinishedWork[-2]))