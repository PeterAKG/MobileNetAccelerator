# This is the simulator to end all simulators... will rewrite from scratch. Want to support:
# Memory
# Squeeze and Excite
# And I want to make sure the order everything is being run in is actually intelligent

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Union
from collections import deque
from typing import List

df = pd.read_excel('MobileNetV3Calc.xlsx')

NUM_MISC_LAYERS = 4

layers = 39
layerList = np.zeros(layers, dtype="U20")
kernelSize = np.zeros(layers, dtype=int)
stride = np.zeros(layers, dtype=int)
channelsIn = np.zeros(layers, dtype=int)
channelsOut = np.zeros(layers, dtype=int)
spatialSize = np.zeros(layers, dtype=int)
squeezeExcite = np.zeros(layers, dtype=int)
SEhiddenLayer = np.zeros(layers, dtype=int)
SRAMStartIndex = np.zeros(layers, dtype=int)
clkCycles = 0

clockHz = 300000000 # 300 MHZ Clock speed
convType = Literal["depthwise", "pointwise", "fullyConnected"]
ActivationType = Literal["depthwise", "pointwise"]

for idx, row in df.iterrows():
    if(idx >= 0 and idx <= 37):
        layerType = row["Layer Type"]
        if(layerType == "fully connected"):
            layerType = "fullyConnected"
        layerList[idx] = layerType
        kernelSize[idx] = int(row["K"])
        stride[idx] = int(row["Stride"])
        channelsIn[idx] = int(row["In"])
        channelsOut[idx] = int(row["Out"])
        spatialSize[idx] = int(row["Spatial"])
        if(row["SE"] == "Yes"):
            squeezeExcite[idx] = 1
            SEhiddenLayer[idx] = int(row["SE Hidden Layer"])
        else:
            squeezeExcite[idx] = 0
        SRAMStartIndex[idx] = int(row["SRAM Activation Start Index"])


DEPTHWISE_KERNEL_PARTS = 9
POINTWISE_KERNEL_PARTS = 9

@dataclass
class DepthwiseWeight:
    type: Literal["depthwise"] = field(default="depthwise", init=False)
    layer: int = -1
    channel: int = -1
    part: int = -1

@dataclass
class DepthwiseKernel:
    type: Literal["depthwise"] = field(default="depthwise", init=False)
    layer: int = -1
    channel: int = -1

@dataclass
class PointwiseWeight:
    type: Literal["pointwise"] = field(default="pointwise", init=False)
    layer: int = -1
    y: int = -1
    channel: int = -1

@dataclass
class PointwiseKernel:
    type: Literal["pointwise"] = field(default="pointwise", init=False)
    layer: int = -1
    channel: int = -1 # This is the output channel
    channelGroup: int = -1 # This is the input channel chunk

@dataclass
class SEWeight:
    type: Literal["SEWeight"] = field(default="SEWeight", init=False)
    matrix: Literal["squeeze", "excite"]
    layer: int = -1
    row: int = -1
    column: int = -1

@dataclass
class FullyConnectedWeight:
    type: Literal["fullyConnected"] = field(default="fullyConnected", init=False)
    layer: int = -1
    row: int = -1
    column: int = -1

Weight = Union[
    DepthwiseWeight,
    PointwiseWeight,
    SEWeight,
    FullyConnectedWeight,
]

Kernel = Union[
    DepthwiseKernel,
    PointwiseKernel
]

# Channel argument refers to the input channel
@dataclass
class Activation:
    type: ActivationType
    layer: int
    x: int
    y: int
    channel: int
    
@dataclass
class ActivationChunk:
    type: ActivationType
    layer: int = -1
    x: int = -1
    y: int = -1
    channel: int = -1
    channelChunk: int = -1   # For pointwise it represents the lowest channel. 

@dataclass
class PartialActivation:
    type: Literal["pointwise"]
    layers: List[int]
    x: int
    y: int
    segment: int
    outputChannel: int

@dataclass
class Layer:
    height: int
    width: int
    values: np.ndarray = field(init=False)

    def __post_init__(self):
        self.values = np.zeros((self.width, self.height), dtype=bool)

# layer -- means we go from the input channel of layer X to output channel of layer X
@dataclass
class DepthwiseConvOperation:
    type: Literal["depthwise"]
    layer: int
    tag: int
    stride: int
    size: int
    flexBuffer: int
    step: int = 0

@dataclass
class CameraDepthwiseConvOperation:
    type: Literal["depthwise"] = field(default="depthwise", init=False)
    tag: int
    targetX: int
    targetY: int
    nextToLoad: int = 0
    nextToRead: int = 0
    nextToTrigger: int = 0
    activationsLoaded: bool = False

@dataclass
class PointwiseConvOperation:
    type: Literal["pointwise"]
    layer: int
    tag: int
    segment: int
    outputChannel: int
    step: int = 0

@dataclass
class FullyConnectedOperation:
    type: Literal["fullyConnected"]
    layer: int
    tag: int
    row: int
    column: int
    step: int = 0

Operation = Union[
    DepthwiseConvOperation,
    CameraDepthwiseConvOperation,
    PointwiseConvOperation,
    FullyConnectedOperation,
]

class SRAM:
    
    def __init__(self):
        self.storage = 18432 # bits
        self.portSize = 18 # bits
        self.addresses = 1024 # Number of available addresses
        self.lastReadClockCycle = -1
        self.lastWriteClockCycle = -1
        
        self.memory = np.empty((self.addresses, SE_WEIGHTS_PER_SRAM_ADDRESS), dtype=object)
        self.memory[:] = None
        

    def isActiveRead(self):
        return self.lastReadClockCycle == clkCycles

    def isActiveWrite(self):
        return self.lastWriteClockCycle == clkCycles

    def readSRAM(self, address):
        if (self.isActiveRead()):
            raise Exception("ERROR! SRAM attempted to read more than once in the same clock cycle")
        self.lastReadClockCycle = clkCycles
        return self.memory[address]

    def writeSRAM(self, address, data):
        if (self.isActiveWrite()):
            raise Exception("ERROR! SRAM attempted to write more than once in the same clock cycle")
        self.lastWriteClockCycle = clkCycles
        self.memory[address] = data
        
class LRAM:
    
    def __init__(self):
        self.storage = 524288 # bits
        self.portSize = 32 # bits
        self.addresses = 16384 # Number of available addresses
        self.lastReadClockCycle = -1
        self.lastWriteClockCycle = -1
        self.memory = np.empty((self.addresses, WEIGHTS_PER_LRAM_ADDRESS), dtype=object)
        self.memory[:] = None
        self.cachedAddr = None
        self.cachedData = None

    def isActiveRead(self):
        return self.lastReadClockCycle == clkCycles

    def isActiveWrite(self):
        return self.lastWriteClockCycle == clkCycles

    # This is not 100% accurate since it is a single cycle read, but it's good enough.
    def readLRAM(self, address):
        if (address == self.cachedAddr):
            return self.memory[address]
        
        if (self.isActiveRead()):
            raise Exception("ERROR! LRAM attempted to read more than once in the same clock cycle")
        self.lastReadClockCycle = clkCycles

        self.cachedAddr = address
        self.cachedData = self.memory[address]
        return self.memory[address]

    def isCached(self, address):
        if (self.cachedAddr == address):
            return True
        else:
            return False

    def writeLRAM(self, address, data):
        if (self.isActiveWrite()):
            raise Exception("ERROR! LRAM attempted to write more than once in the same clock cycle")
        if (len(data) != WEIGHTS_PER_LRAM_ADDRESS):
            raise Exception("ERROR! LRAM writes must contain exactly four weights")
        self.lastWriteClockCycle = clkCycles
        self.memory[address] = data

class flexBuffer:

    def __init__(self):
        self.x = 5
        self.y = 5
        self.z = 8
        self.channel = -1
        
        # These variables keep track of the true x and y val of the top left pixel
        self.trueX = -1
        self.trueY = -1
        
        # This keeps track of how many columns are loaded in, or have been read
        self.numColumnsWritten = 0
        self.numColumnsRead = 0
        
        
    
    # Channel must be // 8
    # Columns are loaded left to right, top to bottom
    def loadColumn(self, x, y, channel, firstColumn):
        if (firstColumn == True):
            self.channel = channel
            self.trueX = x
            self.trueY = y
            self.numColumnsWritten = 1
        else:
            if (channel != self.channel):
                raise Exception("The channel for all columns input must be the same")
            self.numColumnsWritten += 1
            if (self.trueX + (self.numColumnsWritten % self.x) != x) or (self.trueY + (self.numColumnsWritten // self.y) != y):
                raise Exception("flex buffer must be loaded left to right top to bottom")
            
        if (self.numColumnsWritten > self.x * self.y):
            raise Exception("More columns loaded into flex buffer than available")
    
    # Channel is // 8
    # Columns are read left to right, top to bottom
    def readColumn(self):
        if (self.numColumnsRead >= self.numColumnsWritten):
            raise Exception("A column is attempted to be read that has not yet been written into the flex buffer")
        else:            
            # Return format is: x, y, channel // 8
            xToReturn = self.trueX + (self.numColumnsRead % self.x)
            yToReturn = self.trueY + (self.numColumnsRead // self.y)
            toReturn = [xToReturn, yToReturn, self.channel]
            
            self.numColumnsRead += 1
            return toReturn

# I am just going to assume 680 FPS right now with the data coming in consistently at that rate, and adjust once I actually make dev board and solder everything
# and understand the true frame rate
INITIAL_IMAGE_SIZE = 128
class camera:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Tracks what pixel # we have captured
        self.pixelCounter = 0
        
        # When this reaches zero, we can read another pixel. For now, we are going to say 30 system pixels is one pixel clock
        self.clockCounter = 30
        
        self.capturing = False
        
    def startImage(self):
        self.capturing = True
        self.clockCounter = 30
        self.pixelCounter = -1
    
    def updateCamera(self):
        if (self.capturing):
            self.clockCounter -= 1
            if (self.clockCounter <= 0):
                self.clockCounter = 30
                self.pixelCounter += 1
                
            if (self.pixelCounter >= (self.width * self.height - 1)):
                self.capturing = False
    
    def isCapturing(self):
        return self.capturing
    
    def pixelNumber(self):
        return self.pixelCounter

    def pixelCounterToXY(self, pixelCounter):
        assert (pixelCounter >= 0 and pixelCounter < self.width * self.height), "ERROR! Pixel counter is outside the camera image"

        x = pixelCounter % self.width
        y = pixelCounter // self.width
        return [x, y]

    def xyToPixelCounter(self, x, y):
        assert (x >= 0 and x < self.width), "ERROR! Pixel x coordinate is outside the camera width"
        assert (y >= 0 and y < self.height), "ERROR! Pixel y coordinate is outside the camera height"

        return y * self.width + x

    def getActivation(self, x, y):
        assert (x >= 0 and x < self.width), "ERROR! Activation x coordinate is outside the camera width"
        assert (y >= 0 and y < self.height), "ERROR! Activation y coordinate is outside the camera height"

        pixelIndex = self.xyToPixelCounter(x, y)
        assert (self.pixelCounter >= pixelIndex), f"ERROR! Camera has not captured enough pixels for this activation to exist. pixelCounter={self.pixelCounter}, pixelIndex={pixelIndex}"

        return Activation("depthwise", 0, x, y, 0)

imxCamera = camera(INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE)

class DSPunit:
    def __init__(self, dspID):
        self.dspID = dspID

        self.kernel = DepthwiseKernel()
        
        self.preloadedKernel = DepthwiseKernel()
        
        self.activationChunk = ActivationChunk(type="Depthwise")
        
        self.preloadedActivationChunk = ActivationChunk(type="Depthwise")
        
        # Amount of time it takes until it has multiplied all together. When self.timer == 0, it is finished
        self.timer = -1
        self.lastTimerUpdateClockCycle = clkCycles
        
        # The channels that this DSP slice has accumulated
        self.accumulatedChannels = []
        # The x and y coordinates of the accumulated channels
        self.accumulatedX = -1
        self.accumulatedY = -1
        
        # Operation tag -- used for the DSPmanager to mark which operations correspond to which DSPunits
        self.operationTag = None
        
    def preloadKernel(self, kernel: Kernel):        
        self.preloadedKernel = kernel
    
    def loadKernel(self):
        self.kernel = replace(self.preloadedKernel)
        if (isinstance(self.kernel, PointwiseKernel)):
            print(f"""kernel:
                channel: {self.kernel.channelGroup}
                layer: {self.kernel.layer}
                type: {self.kernel.type}""")
        else:
            print(f"""kernel:
                channel: {self.kernel.channel}
                layer: {self.kernel.layer}
                type: {self.kernel.type}""")
    
    def preloadActivation(self, activationChunk: ActivationChunk):
        self.preloadedActivationChunk = activationChunk
        
    def loadActivation(self):
        self.activationChunk = replace(self.preloadedActivationChunk)
        print(f"""activationChunk:
              channel: {self.activationChunk.channel}
              layer: {self.activationChunk.layer}
              type: {self.activationChunk.type}""")
        
    def startDotProduct(self):
        if (isinstance(self.kernel, PointwiseKernel)):
            expectedActivationChannel = self.kernel.channelGroup * POINTWISE_KERNEL_PARTS
        else:
            expectedActivationChannel = self.kernel.channel

        if(self.activationChunk.layer != self.kernel.layer or self.activationChunk.channel != expectedActivationChannel or self.activationChunk.type != self.kernel.type):
            if (self.activationChunk.layer != self.kernel.layer):
                print(f"Mismatched layer! Activation layer: {self.activationChunk.layer}, Kernel layer: {self.kernel.layer}\n")
            if (self.activationChunk.channel != expectedActivationChannel):
                print(f"Mismatched channel! Activation channel: {self.activationChunk.channel}, kernel channel: {expectedActivationChannel}\n")
            if (self.activationChunk.type != self.kernel.type):
                print(f"Mismatched type! Activation type: {self.activationChunk.type}, kernel type: {self.kernel.type}\n")
            raise Exception("ERROR! DSP slice started with mismatched weight and activation")
        elif (self.isActive()):
            raise Exception("ERROR! DSP Slice attempted to start while already being active!")
        else:
            self.timer = 8
            self.lastTimerUpdateClockCycle = clkCycles
            
            if(self.kernel.type == "pointwise"):
                if (len(self.accumulatedChannels) == 0):
                    self.accumulatedX = self.activationChunk.x
                    self.accumulatedY = self.activationChunk.y
                else:
                    if (self.accumulatedX != self.activationChunk.x or self.accumulatedY != self.activationChunk.y):
                        raise Exception("ERROR! DSP slice accumulating at different coordinate than where it started")
                self.accumulatedChannels.append(self.kernel.channelGroup)
            else:
                if (len(self.accumulatedChannels) != 0):
                    raise Exception("Accummulator not cleared before depthwise convolution started")
    
    def updateTimer(self):
        if(self.timer > 0):
            elapsedClockCycles = max(0, clkCycles - self.lastTimerUpdateClockCycle)
            self.timer = max(0, self.timer - elapsedClockCycles)
            self.lastTimerUpdateClockCycle = clkCycles
    
    def isActive(self):
        self.updateTimer()
        return (self.timer > 0)
    
    def getOutput(self):
        if (self.isActive()):
            raise Exception("ERROR! Activation was attempted to be taken from active DSP slice")
        else:
            if (self.kernel.type == "depthwise"):
                toReturn = Activation(
                    type="depthwise",
                    layer=self.kernel.layer,
                    x=self.activationChunk.x,
                    y=self.activationChunk.y,
                    channel=self.kernel.channel)
            else:
                accumulatedChannelSet = set(self.accumulatedChannels)
                usedAllChannels = len(accumulatedChannelSet) == channelsIn[self.kernel.layer]

                if (usedAllChannels):
                    toReturn = Activation(
                        type="pointwise",
                        layer=self.kernel.layer,
                        x=self.accumulatedX,
                        y=self.accumulatedY,
                        channel=self.kernel.channel)
                else:
                    toReturn = PartialActivation(
                        layers=[self.kernel.layer],
                        x=self.accumulatedX,
                        y=self.accumulatedY,
                        segment=self.activationChunk.channelChunk,
                        outputChannel=self.kernel.channel)

                self.accumulatedChannels.clear()
                self.accumulatedX = -1
                self.accumulatedY = -1

            return toReturn
    
    def setOperationTag(self, operationTag):
        self.operationTag = operationTag
    
    def getOperationTag(self):
        return self.operationTag
    
    def getID(self):
        return self.dspID
    
    def getKernel(self):
        return self.kernel

NUM_DSPS = 292
CAMERA_DWC_K = 3
class DSPmanager:
    def __init__(self):
        self.DSPlist = []
        for i in range(NUM_DSPS):
            newDSP = DSPunit(i)

            # A dsp unit and a number indicating whether it has been used this cycle
            self.DSPlist.append(newDSP)
        
        self.operationFIFO = deque()
        
        self.tagCounter = 0
    
    def getFreeDSPunits(self, max=None):
        freeDSPs = [dsp for dsp in self.DSPlist if (not dsp.isActive() and dsp.getOperationTag() is None)]
        if (max is None):
            return freeDSPs
        if (max < 0):
            raise Exception("ERROR! Maximum number of free DSP units cannot be negative")
        return freeDSPs[:max]
    
    def getDSPByID(self, id):
        for dsp in self.DSPlist:
            if (dsp.getID() == id):
                return dsp
    
    def assignOperationTag(self, DSPunits, operationTag):
        for dsp in DSPunits:
            dsp.setOperationTag(operationTag)

    def getDSPsWithOpTag(self, operationTag):
        return [dsp for dsp in self.DSPlist if dsp.getOperationTag() == operationTag]
    
    def fifoPush(self, operation: Operation):
        self.operationFIFO.appendleft(operation)
    
    def fifoPop(self) -> Operation:
        return self.operationFIFO.pop()
    
    def fifoPeek(self) -> Operation:
        return self.operationFIFO[-1]
    
    def fifoCheckLen(self) -> int:
        return len(self.operationFIFO)
    
    def startDepthwiseConvFromCamera(self, targetX, targetY, camera: camera):
        lastNeededX = min(targetX + 1, camera.width - 1)
        lastNeededY = min(targetY + 1, camera.height - 1)
        lastNeededPixel = camera.xyToPixelCounter(lastNeededX, lastNeededY)

        if (camera.pixelNumber() < lastNeededPixel):
            raise Exception("ERROR! Camera pixel counter not advanced enough to do depthwise convolution")
        
        # Since layer 0 is where the camera output gets turned to 16 channels
        freeDSPs = self.getFreeDSPunits(channelsOut[0])
        
        if (len(freeDSPs) != channelsOut[0]):
            return False
        
        self.assignOperationTag(freeDSPs, self.tagCounter)
        depthwiseConvOp = CameraDepthwiseConvOperation(tag=self.tagCounter, targetX=targetX, targetY=targetY)
        self.tagCounter += 1
        
        self.fifoPush(depthwiseConvOp)
        
        return True

    # I am going to go to the gym now, but what I need to do next is have a function which updates the DSPmanager,
    # meaning that the next operation is performed / updated. And then discarded when finished. That same function
    # is also going to need to do further processing on the operation so that it knows exactly what to do with each DSP
    # like... I don't include all necessary info per operation. I expect some processing will need to be done for each
    # instruction
    def processOperation(self):
        if (self.fifoCheckLen() == 0):
            return

        operation = self.fifoPeek()
        
        # Depthwise convolutions go left to right, top to bottom
        INITIAL_CONVOLUTION_K = 3
        if isinstance(operation, CameraDepthwiseConvOperation):
            cameraActivationList = []
            for height in range(INITIAL_CONVOLUTION_K):
                for width in range(INITIAL_CONVOLUTION_K):
                    activationX = operation.targetX + (width - 1)
                    activationY = operation.targetY + (height - 1)
                    if (activationX < 0 or activationX >= imxCamera.width):
                        continue
                    if (activationY < 0 or activationY >= imxCamera.height):
                        continue
                    cameraActivation = imxCamera.getActivation(activationX, activationY)
                    cameraActivationList.append(cameraActivation)
            activationChunk = memory.activationToActivationChunk(cameraActivationList)
            
            DSPunits = self.getDSPsWithOpTag(operation.tag)
            
            if (operation.activationsLoaded == False):
                channel = 0
                for DSPunit in DSPunits:
                    DSPActivationChunk = replace(activationChunk, channel=channel)
                    DSPunit.preloadActivation(DSPActivationChunk)
                    channel += 1
                operation.activationsLoaded = True
            
            if (len(DSPunits) != channelsOut[0]):
                raise Exception("Camera Depthwise Conv Op attempted without 18 DSP units")
            
            if (operation.nextToRead == channelsOut[0]):
                for DSPunit in DSPunits:
                    DSPunit.setOperationTag(None)
                self.fifoPop()
                return
            
            if (operation.nextToRead < operation.nextToTrigger):
                for channel in range(operation.nextToRead, operation.nextToTrigger):
                    DSPtoRead = DSPunits[channel]
                    
                    if (DSPtoRead.isActive() == True):
                        break
                    
                    if (DSPtoRead.isActive() == False):
                        # Load activation from DSP to SRAM
                        if (memory.loadActivationFromDSP(DSPtoRead) == False):
                            break
                        
                        operation.nextToRead += 1
            
            if (operation.nextToTrigger < operation.nextToLoad):
                for channel in range(operation.nextToTrigger, operation.nextToLoad):
                    DSPtoTrigger = DSPunits[channel]
                    
                    if (DSPtoTrigger.isActive()):
                        break
                    
                    DSPtoTrigger.loadKernel()
                    DSPtoTrigger.loadActivation()
                    DSPtoTrigger.startDotProduct()
                    operation.nextToTrigger += 1
            
            if (operation.nextToLoad != channelsOut[0]):
                for channel in range(operation.nextToLoad, channelsOut[0]):
                    
                    desiredKernel = DepthwiseKernel(layer=0, channel=channel)
                    
                    result = memory.askForKernel(desiredKernel, DSPunits[channel].getID())
                    
                    if (result == False):
                        break
                    
                    operation.nextToLoad += 1
    
    def update(self):
        self.processOperation()

DSPMngr = DSPmanager()


NUM_LRAMS = 5
NUM_SRAMS = 166
MAX_SIMULTANEOUS_LRAM_WRITES = NUM_LRAMS
MAX_SIMULTANEOUS_SRAM_WRITES = NUM_SRAMS

WEIGHTS_PER_LRAM_ADDRESS = 4
SE_WEIGHTS_PER_SRAM_ADDRESS = 2
class memoryManager:
    
    #SRAM will be used for activation, and LRAM for weight
    def __init__(self):
        
        self.allSRAM = []
        self.numSRAM = NUM_SRAMS

        for idx in range(self.numSRAM):
            self.allSRAM.append(SRAM())

        self.allLRAM = []
        self.numLRAM = NUM_LRAMS

        for idx in range(self.numLRAM):
            self.allLRAM.append(LRAM())
        
        # This is for loading the weights
        self.weightIndexCount = 0
        self.weightPack = []
        self.SEWeightIndexCount = 0
        self.SEWeightPack = []
        
        # This is for finding weights
        self.startOfLayer = np.zeros((layers, 3), dtype=int)
        self.startOfSELayer = np.zeros((layers, 3), dtype=int)
    
    # This function takes a list of DSPunits and a list of weights, and loads the DSP units with weights as efficiently as possible
    def loadWeightsToDSP(self, DSPunits, weightList):
        
        if (len(DSPunits) != len(weightList)):
            raise Exception("ERROR! Number of DSP slices and weights do not match")

        # Now, go sequentially and load each DSPunit with the corresponding weights
    
    def loadWeightsInStart(self, weight: Weight):
        # Pack four 8-bit weights into one 32-bit LRAM address.
        self.weightPack.append(weight)
        if (len(self.weightPack) < WEIGHTS_PER_LRAM_ADDRESS):
            return

        # Load the packed weights into LRAM
        # weightIndexCount % numLRAM
        # at address
        # weightIndexCount // numLRAM
        
        lramNum = self.weightIndexCount % self.numLRAM
        lramAddr = self.weightIndexCount // self.numLRAM
        
        self.allLRAM[lramNum].writeLRAM(lramAddr, self.weightPack.copy())
        
        self.weightIndexCount += 1
        self.weightPack.clear()

    def loadSEWeightsInStart(self, weight: SEWeight):
        # Pack two squeeze/excite weights into one SRAM address.
        self.SEWeightPack.append(weight)
        if (len(self.SEWeightPack) < SE_WEIGHTS_PER_SRAM_ADDRESS):
            return

        sramNum = self.SEWeightIndexCount % self.numSRAM
        sramAddr = self.SEWeightIndexCount // self.numSRAM

        self.allSRAM[sramNum].writeSRAM(sramAddr, self.SEWeightPack.copy())

        self.SEWeightIndexCount += 1
        self.SEWeightPack.clear()
    
    # Returns the next available LRAM, address, and packed position while loading.
    def getLRAMIndexLoading(self) -> tuple[int, int, int]:
        lramNum = self.weightIndexCount % self.numLRAM
        lramAddr = self.weightIndexCount // self.numLRAM
        packedPosition = len(self.weightPack)
        return (lramNum, lramAddr, packedPosition)

    def markLayerStart(self, layer):
        layerStartPosition = self.getLRAMIndexLoading()
        self.startOfLayer[layer][0] = layerStartPosition[0] # LRAM Num
        self.startOfLayer[layer][1] = layerStartPosition[1] # LRAM Addr
        self.startOfLayer[layer][2] = layerStartPosition[2] # Packed Position

    # Returns the next available SRAM, address, and packed position while loading.
    def getSRAMIndexLoading(self) -> tuple[int, int, int]:
        sramNum = self.SEWeightIndexCount % self.numSRAM
        sramAddr = self.SEWeightIndexCount // self.numSRAM
        packedPosition = len(self.SEWeightPack)
        return (sramNum, sramAddr, packedPosition)

    def markSELayerStart(self, layer):
        layerStartPosition = self.getSRAMIndexLoading()
        self.startOfSELayer[layer][0] = layerStartPosition[0] # SRAM Num
        self.startOfSELayer[layer][1] = layerStartPosition[1] # SRAM Addr
        self.startOfSELayer[layer][2] = layerStartPosition[2] # Packed Position

    def flushWeightsInStart(self):
        if (len(self.weightPack) == 0):
            return

        while (len(self.weightPack) < WEIGHTS_PER_LRAM_ADDRESS):
            self.weightPack.append(None)

        lramNum = self.weightIndexCount % self.numLRAM
        lramAddr = self.weightIndexCount // self.numLRAM
        
        self.allLRAM[lramNum].writeLRAM(lramAddr, self.weightPack.copy())
        
        self.weightIndexCount += 1
        self.weightPack.clear()

    def flushSEWeightsInStart(self):
        if (len(self.SEWeightPack) == 0):
            return

        while (len(self.SEWeightPack) < SE_WEIGHTS_PER_SRAM_ADDRESS):
            self.SEWeightPack.append(None)

        sramNum = self.SEWeightIndexCount % self.numSRAM
        sramAddr = self.SEWeightIndexCount // self.numSRAM

        self.allSRAM[sramNum].writeSRAM(sramAddr, self.SEWeightPack.copy())

        self.SEWeightIndexCount += 1
        self.SEWeightPack.clear()
    
    # Read from (LRAMNum, LRAMAddr, packedIndex), numReads sequentially
    def readSequentially(self, LRAMNum, LRAMAddr, packedIndex, numReads, weightsRead):
        if (packedIndex < 0 or packedIndex >= WEIGHTS_PER_LRAM_ADDRESS):
            raise Exception("ERROR! packedIndex is outside the LRAM packed address")
        if (numReads < 0):
            raise Exception("ERROR! numReads cannot be negative")
        
        currentLRAMNum = LRAMNum
        currentAddr = LRAMAddr
        currentIndex = packedIndex
        
        while (len(weightsRead) < numReads):
            LRAMobj = self.allLRAM[currentLRAMNum]
            
            if (LRAMobj.isCached(currentAddr)):
                fullRead = LRAMobj.readLRAM(currentAddr)
            else:
                if (LRAMobj.isActiveRead()):
                    return False
                else:
                    fullRead = LRAMobj.readLRAM(currentAddr)
            
            while (currentIndex < WEIGHTS_PER_LRAM_ADDRESS and len(weightsRead) < numReads):
                weightsRead.append(fullRead[currentIndex])
                currentIndex += 1
            currentIndex = 0
            currentLRAMNum += 1
            if (currentLRAMNum >= NUM_LRAMS):
                currentLRAMNum = 0
                currentAddr += 1
                
        return True

    def weightsToKernel(self, weightList):
        if (len(weightList) != POINTWISE_KERNEL_PARTS):
            raise Exception("ERROR! Kernel must be made from exactly 9 weights")
        if (any(weight is None for weight in weightList)):
            raise Exception("ERROR! Cannot make kernel from empty weight")

        firstWeight = weightList[0]

        if (isinstance(firstWeight, DepthwiseWeight)):
            if (firstWeight.part != 0):
                raise Exception("ERROR! Depthwise kernel must start at part 0")

            for idx, weight in enumerate(weightList):
                if (not isinstance(weight, DepthwiseWeight)):
                    raise Exception("ERROR! Cannot make kernel from mixed weight types")
                if (weight.layer != firstWeight.layer or weight.channel != firstWeight.channel):
                    raise Exception("ERROR! Depthwise weights are not in the same kernel")
                if (weight.part != idx):
                    raise Exception("ERROR! Depthwise weights are not sequential")

            return DepthwiseKernel(layer=firstWeight.layer, channel=firstWeight.channel)

        if (isinstance(firstWeight, PointwiseWeight)):
            if (firstWeight.channel % POINTWISE_KERNEL_PARTS != 0):
                raise Exception("ERROR! Pointwise kernel must start at an input channel divisible by 9")

            for idx, weight in enumerate(weightList):
                if (not isinstance(weight, PointwiseWeight)):
                    raise Exception("ERROR! Cannot make kernel from mixed weight types")
                if (weight.layer != firstWeight.layer or weight.y != firstWeight.y):
                    raise Exception("ERROR! Pointwise weights are not in the same kernel")
                if (weight.channel != firstWeight.channel + idx):
                    raise Exception("ERROR! Pointwise weights are not sequential")

            return PointwiseKernel(
                layer=firstWeight.layer,
                channel=firstWeight.y,
                channelGroup=firstWeight.channel // POINTWISE_KERNEL_PARTS)

        raise Exception("ERROR! Unsupported weight type for kernel conversion")
    
    def activationToActivationChunk(self, activationList: List[Activation]) -> ActivationChunk:
        if (len(activationList) == 0):
            raise Exception("ERROR! Cannot make activation chunk from empty activation list")
        if (any(activation is None for activation in activationList)):
            raise Exception("ERROR! Cannot make activation chunk from empty activation")

        firstActivation = activationList[0]
        if (any(activation.type != firstActivation.type for activation in activationList)):
            raise Exception("ERROR! Cannot make activation chunk from mixed activation types")

        if (firstActivation.type == "depthwise"):
            layerKernelSize = kernelSize[firstActivation.layer]
            kernelRadius = layerKernelSize // 2
            layerSpatialSize = spatialSize[firstActivation.layer]

            if (layerKernelSize != 3 and layerKernelSize != 5):
                raise Exception("ERROR! Unsupported depthwise activation kernel size")

            for activation in activationList:
                if (activation.layer != firstActivation.layer or activation.channel != firstActivation.channel):
                    raise Exception("ERROR! Depthwise activations are not in the same chunk")
                if (activation.x < 0 or activation.x >= layerSpatialSize):
                    raise Exception("ERROR! Depthwise activation x coordinate is outside the layer bounds")
                if (activation.y < 0 or activation.y >= layerSpatialSize):
                    raise Exception("ERROR! Depthwise activation y coordinate is outside the layer bounds")

            activationCoordinates = [(activation.x, activation.y) for activation in activationList]
            possibleTopLeftCoordinates = []
            for topLeftY in range(firstActivation.y - layerKernelSize + 1, firstActivation.y + 1):
                for topLeftX in range(firstActivation.x - layerKernelSize + 1, firstActivation.x + 1):
                    expectedCoordinates = []
                    for kernelY in range(layerKernelSize):
                        for kernelX in range(layerKernelSize):
                            expectedX = topLeftX + kernelX
                            expectedY = topLeftY + kernelY
                            if (expectedX < 0 or expectedX >= layerSpatialSize):
                                continue
                            if (expectedY < 0 or expectedY >= layerSpatialSize):
                                continue
                            expectedCoordinates.append((expectedX, expectedY))

                    if (activationCoordinates == expectedCoordinates):
                        possibleTopLeftCoordinates.append((topLeftX, topLeftY))

            if (len(possibleTopLeftCoordinates) != 1):
                raise Exception("ERROR! Depthwise activations are not in clipped row-major kernel order")

            topLeftX = possibleTopLeftCoordinates[0][0]
            topLeftY = possibleTopLeftCoordinates[0][1]

            return ActivationChunk(
                type=firstActivation.type,
                layer=firstActivation.layer,
                x=topLeftX + kernelRadius,
                y=topLeftY + kernelRadius,
                channel=firstActivation.channel)

        if (firstActivation.type == "pointwise"):
            if (len(activationList) != POINTWISE_KERNEL_PARTS):
                raise Exception("ERROR! Pointwise activation chunk has incorrect number of activations")

            lowestChannel = min(activation.channel for activation in activationList)
            expectedChannels = set(range(lowestChannel, lowestChannel + len(activationList)))
            actualChannels = {activation.channel for activation in activationList}
            if (actualChannels != expectedChannels):
                raise Exception("ERROR! Pointwise activations are not sequential")

            for activation in activationList:
                if (activation.layer != firstActivation.layer):
                    raise Exception("ERROR! Pointwise activations are not in the same layer")
                if (activation.x != firstActivation.x or activation.y != firstActivation.y):
                    raise Exception("ERROR! Pointwise activations are not at the same coordinate")

            return ActivationChunk(
                type=firstActivation.type,
                layer=firstActivation.layer,
                x=firstActivation.x,
                y=firstActivation.y,
                channel=lowestChannel,
                channelChunk=lowestChannel)

        raise Exception("ERROR! Unsupported activation type for activation chunk conversion")
    

    def askForKernel(self, kernel: Kernel, id):
        # We basically need to bridge the gap between the start of the layer and the
        # position of the kernel just by knowing the
        # layer and the channel (and channel group for pointwise)
        
        if isinstance(kernel, DepthwiseKernel):
            totalSteps = kernel.channel * DEPTHWISE_KERNEL_PARTS     # Total number of weights forward we need to go
            
            # There is a fixed divisor hardware which only uses multiplies and shifts
            
            layerLRAMStart = self.startOfLayer[kernel.layer][0]
            layerAddrStart = self.startOfLayer[kernel.layer][1]
            layerPackedStart = self.startOfLayer[kernel.layer][2]
            
            LRAMsteps = totalSteps // WEIGHTS_PER_LRAM_ADDRESS
            packedJumps = totalSteps % WEIGHTS_PER_LRAM_ADDRESS
            
            # I am writing it in this dumbass way to give you an idea how to do it in hardware
            logicalLRAM = layerLRAMStart + LRAMsteps
            LRAMKernelStart = (logicalLRAM % NUM_LRAMS)
            AddrKernelStart = layerAddrStart + (logicalLRAM // NUM_LRAMS)
            
            packedKernelStart = layerPackedStart + packedJumps
            if (packedKernelStart >= WEIGHTS_PER_LRAM_ADDRESS):
                packedKernelStart -= WEIGHTS_PER_LRAM_ADDRESS
                LRAMKernelStart += 1
                if (LRAMKernelStart >= NUM_LRAMS):
                    LRAMKernelStart -= NUM_LRAMS
                    AddrKernelStart += 1
            
            # So, now we know that the kernel starts at:
            # (LRAMKernelStart, AddrKernelStart, packedKernelStart)
            # We just need to read the following 9 weights from memory, and feed it to a DSP unit
            
            # This shouldn't error out when you are too greedy with your reads, but it does. The readSequentially
            # function should take in a list as an object or return a tuple with a list, and also return true/false
            # depending on if the read was successful.
            weights = []
            success = self.readSequentially(LRAMKernelStart, AddrKernelStart, packedKernelStart, DEPTHWISE_KERNEL_PARTS, weights)
            if (success == False):
                return False
            
            desiredKernel = kernel
            returnedKernel = self.weightsToKernel(weights)
            if (returnedKernel != desiredKernel):
                raise Exception(f"ERROR! Memory returned the wrong kernel. Desired kernel: {desiredKernel}, returned kernel: {returnedKernel}")
            # Get DSPUnit by ID
            # Load kernel to DSPUnit
            
            dsp = DSPMngr.getDSPByID(id)
            
            dsp.preloadKernel(returnedKernel)
        elif (isinstance(kernel, PointwiseKernel)):
            # I will implement this later, should be similar to code above, just need to add a check
            # for if it is at the part of the code where we store one half of a chunk near another
            return False
        else:
            raise Exception("Unrecognized Kernel type")
        
        return True
    
    def loadActivationsFromDSPs(self, DSPWithActivation, DSPWithActivation2):
        if (DSPWithActivation.isActive()):
            raise Exception("Attempted to load activation to memory from an active DSP")
        if (DSPWithActivation2.isActive()):
            raise Exception("Attempted to load activation to memory from an active DSP")

        # Now, I need to come up with a scheme on how activations should be loaded and fetched from memory
        activation1 = DSPWithActivation.getOutput()
        activation2 = DSPWithActivation2.getOutput()
        
        if (activation1.layer != activation2.layer):
            raise Exception("Activations do not come from the same layer!")
        
        if (layerList[activation1.layer + 1] == "depthwise"):
            pass
        elif (layerList[activation1.layer + 1] == "pointwise"):
            return self.loadActivationsToSRAM(activation1, activation2)
        else:
            raise Exception("Loading activation from DSP to non pointwise or depthwise layers not supported... for now")

    def loadActivationFromDSP(self, DSPWithActivation):
        if (DSPWithActivation.isActive()):
            raise Exception("Attempted to load activation to memory from an active DSP")

        activation = DSPWithActivation.getOutput()

        if (layerList[activation.layer + 1] == "depthwise"):
            pass
        elif (layerList[activation.layer + 1] == "pointwise"):
            return self.loadActivationToSRAM(activation)
        else:
            raise Exception("Loading activation from DSP to non pointwise or depthwise layers not supported... for now")

    def getActivationSRAMLocation(self, activation):
        layer = activation.layer
        numChannels = channelsOut[layer]
        maxX = spatialSize[layer]

        if (activation.channel < 0 or activation.channel >= numChannels):
            raise Exception("ERROR! Activation channel is outside the layer channel range")
        if (activation.x < 0 or activation.x >= maxX):
            raise Exception("ERROR! Activation x coordinate is outside the layer spatial range")
        if (activation.y < 0 or activation.y >= maxX):
            raise Exception("ERROR! Activation y coordinate is outside the layer spatial range")

        index = activation.channel + (activation.x * numChannels) + (maxX * numChannels * activation.y)
        trueIndex = index + SRAMStartIndex[layer]
        packedIndex = trueIndex % SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMJumps = trueIndex // SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMNum = SRAMJumps % NUM_SRAMS
        SRAMAddr = SRAMJumps // NUM_SRAMS

        return (SRAMNum, SRAMAddr, packedIndex)
    
    def loadActivationsToSRAM(self, activation1, activation2):
        if (activation1.layer != activation2.layer):
            raise Exception("ERROR! Activations must be from the same layer to pack into one SRAM address")

        activation1Location = self.getActivationSRAMLocation(activation1)
        activation2Location = self.getActivationSRAMLocation(activation2)

        if (activation1Location[0] != activation2Location[0] or activation1Location[1] != activation2Location[1]):
            raise Exception("ERROR! Activations must pack into the same SRAM and address")
        if (activation1Location[2] == activation2Location[2]):
            raise Exception("ERROR! Activations cannot use the same packed SRAM position")

        SRAMNum = activation1Location[0]
        SRAMAddr = activation1Location[1]
        SRAMobj = self.allSRAM[SRAMNum]

        if (SRAMobj.isActiveWrite()):
            return False

        packedActivations = [None] * SE_WEIGHTS_PER_SRAM_ADDRESS
        packedActivations[activation1Location[2]] = activation1
        packedActivations[activation2Location[2]] = activation2
        SRAMobj.writeSRAM(SRAMAddr, packedActivations)

        return True

    def loadActivationToSRAM(self, activation):
        activationLocation = self.getActivationSRAMLocation(activation)

        SRAMNum = activationLocation[0]
        SRAMAddr = activationLocation[1]
        packedIndex = activationLocation[2]
        SRAMobj = self.allSRAM[SRAMNum]

        if (SRAMobj.isActiveRead() or SRAMobj.isActiveWrite()):
            return False

        packedActivations = list(SRAMobj.readSRAM(SRAMAddr))
        packedActivations[packedIndex] = activation
        SRAMobj.writeSRAM(SRAMAddr, packedActivations)

        return True
            
            
            
            

# Before all else, load the LRAMs with their corresponding weights using the encoding scheme I invented:
# Remember, 5 LRAM's are going to store all weights in the order that they are accessed, so the weights
# will be filled like this:
# | LRAM 1 | LRAM 2 | LRAM 3 | LRAM 4 | LRAM 5 |
# |   w0   |   w1   |   w2   |   w3   |   w4   |
# |   w5   |   w6   |   w7   |   w8   |   w9   |
# |  ...   |  ...   |  ...   |  ...   |  ...   |
# Where w0 is accessed 1st, then w1, and so on

layer = 0
indexPointer = 0
memory = memoryManager()

# NEW FORMAT?
# Memory format:
# IF POINTWISE:
# If the number of channels and the spatial size permits us to calculate 1/4th of the kernel independently,
# then the format is:
# Weight 0 - Channel 0, Weight 1 - Channel 0, Weight 2 - Channel 0, Weight 3 - Channel 0, Weight 4 - Channel 0
# Weight 5 - Channel 0, Weight 6 - Channel 0, Weight 7 - Channel 0, Weight 8 - Channel 0, 
# Weight 0 - Channel 1, Weight 1 - Channel 1, Weight 2 - Channel 1, ...
# Otherwise, store the weights like this:
# Weight 0 - Channel 0, Weight 1 - Channel 0, Weight 2 - Channel 0, Weight 3 - Channel 0, Weight 4 - Channel 0,
# Weight 5 - Channel 0, Weight 6 - Channel 0, Weight 7 - Channel 0, Weight 8 - Channel 0
# Weight (channelsIn[layer] // 2) - Channel 0, Weight (channelsIn[layer] // 2) + 1 - Channel 0,
# Weight (channelsIn[layer] // 2) + 2 - Channel 0, Weight (channelsIn[layer] // 2) + 3 - Channel 0,
# IF DEPTHWISE:
# Store the weights in blocks of 9, with increasing channels:
# Weight 0 - Channel 0, Weight 1 - Channel 0, Weight 2 - Channel 0, Weight 3 - Channel 0,
# Weight 4 - Channel 0, Weight 5 - Channel 0, Weight 6 - Channel 0, Weight 7 - Channel 0,
# Weight 8 - Channel 0
# I will probably do it this way, but I will only bother trying once I need to

# Memory format:
# IF POINTWISE:
# If the number of channels and the spatial size permits us to calculate 1/4th of the kernel independently,
# then the format is:
# Weight 0 - Channel 0, Weight 0 - Channel 1, Weight 0 - Channel 2, Weight 0 - Channel 3, etc
# Weight 1 - Channel 0, Weight 1 - Channel 1, Weight 1 - Channel 2, Weight 1 - Channel 3, etc
# Otherwise, store the weights like this:
# Weight 0 - Channel 0, Weight (channelsIn[layer] // 2) - Channel 0
# Weight 0 - Channel 1, Weight (channelsIn[layer] // 2) - Channel 1
# etc
# then...
# Weight 1 - Channel 0, Weight (channelsIn[layer] // 2) + 1 - Channel 0
# Weight 1 - Channel 1, Weight (channelsIn[layer] // 2) + 1 - Channel 1
# IF DEPTHWISE:
# Store the weights in blocks of 9, with increasing channels:
# Weight 0 - Channel 0, Weight 1 - Channel 0, Weight 2 - Channel 0, Weight 3 - Channel 0,
# Weight 4 - Channel 0, Weight 5 - Channel 0, Weight 6 - Channel 0, Weight 7 - Channel 0,
# Weight 8 - Channel 0
#
# Also, we should store where the start of every layer is
startOfLayers = []

for layer in range(layers - NUM_MISC_LAYERS):
    memory.markLayerStart(layer) 
    channel = 0
    channelsCompleted = 0
    part = 0
    
    if (layerList[layer] == "depthwise"):
        layerKernelParts = DEPTHWISE_KERNEL_PARTS
        chunkSize = DEPTHWISE_KERNEL_PARTS
    elif (layerList[layer] == "pointwise"):
        layerKernelParts = channelsIn[layer]
        chunkSize = POINTWISE_KERNEL_PARTS
    else:
        raise Exception("ERROR! Unsupported convolution weight type")

    if (((NUM_DSPS//channelsOut[layer]) > (spatialSize[layer] // 4)) and layerList[layer] == "pointwise"):
        outputOffset = 0
        inputOffset = 0
        part = 0
        storeSecondHalf = False
        while outputOffset < (channelsOut[layer]//2):
            for cycle in range(MAX_SIMULTANEOUS_LRAM_WRITES * WEIGHTS_PER_LRAM_ADDRESS):
                if (not storeSecondHalf):
                    outputChannel = outputOffset
                else:
                    outputChannel = outputOffset + (channelsOut[layer]//2)

                inputChannel = part + inputOffset*chunkSize
                weight = PointwiseWeight(layer=layer, channel=inputChannel, y=outputChannel)
                memory.loadWeightsInStart(weight)

                part += 1
                if (part >= chunkSize):
                    inputOffset += 1
                    part = 0

                if (inputOffset*chunkSize >= layerKernelParts):
                    inputOffset = 0
                    part = 0
                    if (storeSecondHalf):
                        outputOffset += 1
                    storeSecondHalf = not storeSecondHalf

                if (outputOffset >= (channelsOut[layer]//2)):
                    break

            clkCycles += 1
    elif (layerList[layer] == "pointwise"):
        outputChannel = 0
        offset = 0
        totalParts = 0
        while outputChannel < channelsOut[layer]:
            for cycle in range(MAX_SIMULTANEOUS_LRAM_WRITES * WEIGHTS_PER_LRAM_ADDRESS):
                totalParts = part + offset*chunkSize
                weight = PointwiseWeight(layer=layer, channel=totalParts, y=outputChannel)
                memory.loadWeightsInStart(weight)

                part += 1
                if (part >= chunkSize):
                    offset += 1
                    part = 0

                if (offset*chunkSize >= layerKernelParts):
                    outputChannel += 1
                    offset = 0
                    part = 0

                if (outputChannel >= channelsOut[layer]):
                    break

            clkCycles += 1
    else:
        offset = 0
        totalParts = 0
        while offset*chunkSize < layerKernelParts:
            for cycle in range(MAX_SIMULTANEOUS_LRAM_WRITES * WEIGHTS_PER_LRAM_ADDRESS):
                if (layerList[layer] == "depthwise"):
                    weight = DepthwiseWeight(layer, channel, part)
                elif (layerList[layer] == "pointwise"):
                    weight = PointwiseWeight(layer=layer, channel=channel,
                        y=totalParts)
                else:
                    raise Exception("ERROR! Unsupported convolution weight type")
                memory.loadWeightsInStart(weight)
                
                part += 1
                if (part >= chunkSize):
                    channel += 1
                    part = 0
                
                if (channel >= channelsIn[layer]):
                    offset += 1
                    channel = 0
                    if (layerList[layer] == "depthwise"):
                        break
                
                totalParts = part+offset*chunkSize
                
                if (offset*chunkSize >= layerKernelParts):
                    break
                
            clkCycles += 1

# For this next part, we are just going to store every row of the matrix sequentially
# So it looks like:
# Row 0 - Col 0, Row 0 - Col 1, Row 0 - Col 2, Row 0 - Col 3, Row 0 - Col 4, Row 0 - Col 5...
# Row 1 - Col 0, Row 1 - Col 1, Row 1 - Col 2, Row 1 - Col 3, Row 1 - Col 4, Row 1 - Col 5...

# + 1 to skip the gap layer
MISC_LAYER_START = layers - NUM_MISC_LAYERS + 1

for misc_layer in range(MISC_LAYER_START, layers):
    memory.markLayerStart(misc_layer)
    row = 0
    column = 0
    while (row < channelsOut[misc_layer]):
        for cycle in range(MAX_SIMULTANEOUS_LRAM_WRITES * WEIGHTS_PER_LRAM_ADDRESS):
            weight = FullyConnectedWeight(misc_layer, row, column)
            column += 1
            if (column >= channelsIn[misc_layer]):
                column = 0
                row += 1
            memory.loadWeightsInStart(weight)
            if (row >= channelsOut[misc_layer]):
                break
        clkCycles += 1

memory.flushWeightsInStart()

# Add here -- for each layer with Squeeze-Excite, store the squeeze excite neural net
# The neural network is composed of two matrices. One of size
# (Output Channel x SE Hidden Layer size)
# one of size
# (SE Hidden Layer size x Output Channel)
# Store both matrices one after another, row after row.
for layer in range(layers - NUM_MISC_LAYERS):
    if (squeezeExcite[layer] != 1):
        continue

    memory.markSELayerStart(layer)

    matrix = "squeeze"
    row = 0
    column = 0
    while (matrix != "done"):
        for cycle in range(MAX_SIMULTANEOUS_SRAM_WRITES * SE_WEIGHTS_PER_SRAM_ADDRESS):
            weight = SEWeight(matrix=matrix, layer=layer, row=row, column=column)
            memory.loadSEWeightsInStart(weight)

            column += 1
            if (matrix == "squeeze"):
                if (column >= SEhiddenLayer[layer]):
                    column = 0
                    row += 1

                if (row >= channelsOut[layer]):
                    matrix = "excite"
                    row = 0
                    column = 0
            elif (matrix == "excite"):
                if (column >= channelsOut[layer]):
                    column = 0
                    row += 1

                if (row >= SEhiddenLayer[layer]):
                    matrix = "done"

            if (matrix == "done"):
                break

        clkCycles += 1

memory.flushSEWeightsInStart()

imxCamera = camera(INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE)

# Below is the behavior of the global controller:

imxCamera.startImage()

pixelsComputed = 0

print("Hello!\n")

while (pixelsComputed < INITIAL_IMAGE_SIZE ** 2):
    imxCamera.updateCamera()
    
    pixelNumber = imxCamera.pixelNumber()
    if (pixelNumber < 0):
        continue

    pixelCoords = imxCamera.pixelCounterToXY(pixelNumber)
    pixelX = pixelCoords[0]
    pixelY = pixelCoords[1]
    
    # This is the next output we want to calculate
    targetOutput = imxCamera.pixelCounterToXY(pixelsComputed)
    
    targetX = targetOutput[0]
    targetY = targetOutput[1]
    
    if (targetY >= pixelY):
        continue
    
    if (targetY + 1 == pixelY):
        if (targetX + 1 > pixelX):
            continue
    
    print(f"targetX: {targetX}, targetY: {targetY}\n")
    if (DSPMngr.startDepthwiseConvFromCamera(targetX, targetY, imxCamera)):
        pixelsComputed += 1
    
    DSPMngr.update()

