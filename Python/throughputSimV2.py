import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

totalTime = 4999200 * 2
df = pd.read_excel('ThroughputCalcV2.xlsx')

totalTime = int(df["Time to output (CLK CYCLES)"][55]) * 4

print("total time is " + str(totalTime))

clkCycles = 0
timeToOutput = []
inputChannelsCompleted = []
inputChannelCompletionTimes = []
inputChannelStartTimes = []
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

counter = 0

for idx, row in df.iterrows():
    if(idx >= 1 and idx < 51):
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
        
        if(layerType[idx - 1] == "Depthwise"):
            inputChannelCompletionTimes.append([])
            inputChannelStartTimes.append([])
            
        if(layerType[idx - 1] == "Pointwise"):
            inputChannelCompletionTimes.append([])
            for channel in range(totalOutputChannels[idx - 1]):
                inputChannelCompletionTimes[idx - 1].append([])
                
            inputChannelStartTimes.append([])
            for channel in range(totalOutputChannels[idx - 1]):
                inputChannelStartTimes[idx - 1].append([])

inputChannelsCompleted.append(0)
totalInputChannels.append(totalOutputChannels[-1])

class BRAMPrimitive:
    
    def __init__(self, storage):
        self.storage = storage
        self.memory = np.array(storage, dtype=np.uint8)
        
        
# class BRAM:
    
#     def __init__(self, storage, throughput):
#         self.storage = storage #This is in bytes, should be 36,864 / 8 * 50 = 230400
#         self.throughput = throughput #This is in operations/clk cycle. AKA, how many reads or writes can we do per cycle

class DWCV:
    
    def __init__(self):
        self.dwcvCounter = 0
        self.dwcvActive = []
        self.layerWorkingOn = -1
        # Format is layer, input channel
        self.pingLayerWeights = (-1, -1)
        self.pongLayerWeights = (-1, -1)
            
    def updateActivity(self):
        if(self.dwcvCounter > 0):
            self.dwcvActive.append(1)
            self.dwcvCounter -= 1
            if(self.dwcvCounter == 0):
                inputChannelsCompleted[self.layerWorkingOn] += 1
                inputChannelCompletionTimes[self.layerWorkingOn - 1].append(clkCycles)
                
                if(inputChannelsCompleted[self.layerWorkingOn] % 32 == 0):
                    dwcvCreated32.append(clkCycles)
                self.layerWorkingOn = -1
        else:
            self.dwcvActive.append(0)
    
    def startWorking(self, clockCycles, layer):
        
        # if(self.getPingLayer() != (layer, inputChannelsUsedDWCV[layer - 1])):
        #     RuntimeError("Error: Depthwise Convolution was attempted to start without kernel loaded")
        #     return 
        
        if(layer == 2):
            print("Something is wrong, and we are at " + str(clkCycles))
            
        self.dwcvCounter = clockCycles
        self.layerWorkingOn = layer
        inputChannelsUsedDWCV[layer - 1] += 1
        inputChannelStartTimes[self.layerWorkingOn - 1].append(clkCycles)
        for idx in range(9):
            itemsToRemove.append((layer - 1, inputChannelsUsedDWCV[layer - 1], 0))
        
    def isActive(self):
        if(self.dwcvCounter > 0):
            return True
        else:
            return False
        
    def getActivity(self):
        return self.dwcvActive
    
    def getPingLayer(self):
        return self.pingLayerWeights
    
    def getPongLayer(self):
        return self.pongLayerWeights
    
    def pingToPong(self):
        self.pongLayerWeights = self.pingLayerWeights
        self.pingLayerWeights = -1
    
    def setPingLayerWeight(self, layer, inputChannel):
        self.pingLayerWeights = (layer, inputChannel)
        
class PWCV:
    
    def __init__(self, numPWCV):
        self.pwcvCounter = 0
        self.pwcvActive = []
        self.layerWorkingOn = -1
        self.numPWCV = numPWCV
        self.inputChannelsWorkingOn = []
        
        # Format is layer, layer/input channel/input channel from previous layer
        # Also, all weights are loaded in groups of 32. So the input channel signifies what block of 32 weights we are
        # doing. The input channel from previous layer signifies which block of 32 we are using there
        # self.pingLayerWeights = (-1, -1, -1)
        # self.pongLayerWeights = (-1, -1, -1)
    
    def updateActivity(self):
        if(self.pwcvCounter > 0):
            self.pwcvActive.append(1)
            self.pwcvCounter -= 1
            if(self.pwcvCounter == 0):
                pwcvFinishedWork.append(clkCycles)
                if(all(inputChannelsUsedPWCV[self.layerWorkingOn - 1][inputChannel] == totalInputChannels[self.layerWorkingOn - 1] for inputChannel in self.inputChannelsWorkingOn)):
                    inputChannelsCompleted[self.layerWorkingOn] += 8 # Since PWCV unit works on batches of 32 in parallel
                    pwcvCreated32.append(clkCycles)
                
                for inputChannel in self.inputChannelsWorkingOn:
                    inputChannelCompletionTimes[self.layerWorkingOn - 1][inputChannel].append(clkCycles)
                
                self.layerWorkingOn = -1
                self.inputChannelsWorkingOn = []
        else:
            self.pwcvActive.append(0)
    
    def startWorking(self, clockCycles, layer, inputChannels):
        # for inputChannel in inputChannels:
        #     if(mem.contains(layer - 1, inputChannel, (inputChannelsUsedPWCV[layer - 1][inputChannel] + 32) // 32) != 32):
        #         RuntimeError("Error: Pointwise Convolution was attempted to start without kernel loaded")
        
        self.pwcvCounter = clockCycles//self.numPWCV
        self.layerWorkingOn = layer
        self.inputChannelsWorkingOn = inputChannels #inputChannels is a list of input channels the PWCV unit is working on concurrently
        
        for inputChannel in inputChannels:
            for idx in range(8):
                itemsToRemove.append((layer, inputChannel, (inputChannelsUsedPWCV[layer - 1][inputChannel] + 8) // 8))
            inputChannelsUsedPWCV[layer - 1][inputChannel] += 8
            inputChannelStartTimes[self.layerWorkingOn - 1][inputChannel].append(clkCycles)
        
        
    def isActive(self):
        if(self.pwcvCounter > 0):
            return True
        else:
            return False
    
    def getActivity(self):
        return self.pwcvActive
    
    def getPingLayer(self):
        return self.pingLayerWeights
    
    def getPongLayer(self):
        return self.pongLayerWeights
    
    def pingToPong(self):
        self.pongLayerWeights = self.pingLayerWeights
        self.pingLayerWeights = -1
    
    def setPingLayerWeight(self, layer, inputChannel1, inputChannel2):
        self.pingLayerWeights = (layer, inputChannel1, inputChannel2)
    

inputChannelsCompleted[0] = totalInputChannels[0] # Should be 32, assume all input channels have been finished by standard convolution
layerWorkingOn = 1
completedLayers = [] # Number of clock cycles at which layers got completed
completedLayers.append(0)

dwcvUnit = DWCV()
pwcvUnits = PWCV(8)

for clkCycles in range(totalTime):
    if(not dwcvUnit.isActive()):
        for idx in range(1, len(inputChannelsCompleted)):
            if(inputChannelsCompleted[idx] < totalInputChannels[idx] and inputChannelsCompleted[idx - 1] > inputChannelsUsedDWCV[idx - 1]):
                if(layerType[idx - 1] == "Depthwise"):
                    dwcvUnit.startWorking((spatialSize[idx - 1] * spatialSize[idx - 1]) // stride[idx-1], idx)
    
    if(not pwcvUnits.isActive()):
        for idx in range(1, len(inputChannelsCompleted)):
            if(inputChannelsCompleted[idx] < totalInputChannels[idx] and layerType[idx - 1] == "Pointwise"):
                for channelBlock in range(totalInputChannels[idx]//8):
                    condition = True
                    channelsWorkingOn = []
                    for channel in range(8):
                        if(inputChannelsUsedPWCV[idx - 1][channel + 8*channelBlock] <= inputChannelsCompleted[idx - 1] - 8):
                            condition = condition and True
                            channelsWorkingOn.append(channel + 8*channelBlock)
                        else:
                            condition = condition and False
                            break
                    if condition is True:
                        # pwcvUnits.startWorking(timeToOutput[idx-1]/(totalInputChannels[idx]/32) * (1/(totalInputChannels[idx-1]/32)), idx, channelsWorkingOn)
                        pwcvUnits.startWorking(8 * (spatialSize[idx - 1] * spatialSize[idx - 1]), idx, channelsWorkingOn)
                        break
                break #We can stop on the first condition this is true

    dwcvUnit.updateActivity()
    pwcvUnits.updateActivity()
    
    if(layerWorkingOn == len(inputChannelsCompleted)):
        print("Layer working on is " + str(layerWorkingOn) + " and len(inputChannelsCompleted) is " + str(len(inputChannelsCompleted)))
        break
    
    try:
        if(inputChannelsCompleted[layerWorkingOn] == totalInputChannels[layerWorkingOn]):
            completedLayers.append(clkCycles)
            layerWorkingOn += 1
    except:
        print("layerWorkingOn is " + str(layerWorkingOn))
            
        
    if clkCycles % 10000 == 0:
        print("Cycle: ", clkCycles)
        print("pwcv.isActive(): ", pwcvUnits.isActive())
        print("dwcv.isActive(): ", dwcvUnit.isActive())
        print("pwcv count: ", pwcvUnits.pwcvCounter)
        print("dwcv count: ", dwcvUnit.dwcvCounter)

print("Layer working on is " + str(layerWorkingOn) + " and len(inputChannelsCompleted) is " + str(len(inputChannelsCompleted)))

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

# for idx in range(len(inputChannelCompletionTimes)):
#     if(layerType[idx] == "Depthwise"):
#         print("Depthwise layer:")
#         for jdx in range (len(inputChannelCompletionTimes[idx])):
#             print(f"Time taken to use output channel {jdx} of layer {idx}: {inputChannelCompletionTimes[idx][jdx]}")
#     if(layerType[idx] == "Pointwise"):
#         print("Pointwise layer:")
#         for jdx in range (len(inputChannelCompletionTimes[idx])):
#             for ic in range (len(inputChannelCompletionTimes[idx][jdx])):
#                 print(f"Time taken to use output channel {ic} of layer {idx} to create input channel {jdx}: {inputChannelCompletionTimes[idx][jdx]}")

import json

# Build a structured object that matches your data
payload = []
for idx in range(len(inputChannelCompletionTimes)):
    lt = layerType[idx]

    if lt == "Depthwise":
        payload.append({
            "layer": idx,
            "type": "Depthwise",
            "res": int(spatialSize[idx]),
            "channels": [
                {
                    "out_ch": jdx,
                    "starts": inputChannelStartTimes[idx][jdx],
                    "ends":   inputChannelCompletionTimes[idx][jdx],
                }
                for jdx in range(len(inputChannelCompletionTimes[idx]))
            ],
        })

    elif lt == "Pointwise":
        payload.append({
            "layer": idx,
            "type": "Pointwise",
            "res": int(spatialSize[idx]),
            "out_channels": [
                {
                    "out_ch": jdx,
                    "in_channels": [
                        {
                            "in_ch": ic,
                            "starts": inputChannelStartTimes[idx][jdx][ic],
                            "ends":   inputChannelCompletionTimes[idx][jdx][ic],
                        }
                        for ic in range(len(inputChannelCompletionTimes[idx][jdx]))
                    ],
                }
                for jdx in range(len(inputChannelCompletionTimes[idx]))
            ],
        })

with open("mobilenet_timeline.json", "w") as f:
    json.dump(payload, f)
