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
        else:
            squeezeExcite[idx] = 0


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
    channel: int = -1
    channelGroup: int = -1 # This is // 9

@dataclass
class FullyConnectedWeight:
    type: Literal["fullyConnected"] = field(default="fullyConnected", init=False)
    layer: int = -1
    row: int = -1
    column: int = -1

Weight = Union[
    DepthwiseWeight,
    PointwiseWeight,
    FullyConnectedWeight,
]

Kernel = Union[
    DepthwiseKernel,
    PointwiseKernel
]
    
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
    channelChunk: int = -1   # This is // 9

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
    type: Literal["depthwise"]
    tag: int
    targetX: int
    targetY: int
    xOffset: int = -1
    yOffset: int = -1
    nextToLoad: int = 0
    nextToRead: int = 0

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
        
        self.memory = np.empty(self.addresses, dtype=[('a', object), ('b', object)])
        self.memory['a'] = None
        self.memory['b'] = None
        

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
        self.pixelCounter = 0
    
    def updateCamera(self):
        if (self.capturing):
            self.clockCounter -= 1
            if (self.clockCounter <= 0):
                self.clockCounter = 30
                self.pixelCounter += 1
                
            if (self.pixelCounter >= self.width * self.height):
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
        assert (self.pixelCounter > pixelIndex), "ERROR! Camera has not captured enough pixels for this activation to exist"

        return Activation("depthwise", 0, x, y, 0)

imxCamera = camera(INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE)

class DSPunit:
    def __init__(self, dspID):
        self.dspID = dspID

        self.kernel = DepthwiseKernel()
        
        self.preloadedKernel = DepthwiseKernel()
        
        self.activationChunk = ActivationChunk()
        
        self.preloadedActivationChunk = ActivationChunk()
        
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
    
    def preloadActivation(self, activationChunk: ActivationChunk):
        self.preloadedActivationChunk = activationChunk
        
    def loadActivation(self):
        self.activationChunk = replace(self.preloadedActivationChunk)
        
    def startDotProduct(self):
        if(self.activationChunk.layer != self.kernel.layer or self.activationChunk.channel != self.kernel.channel or self.activationChunk.type != self.kernel.type):
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
                self.accumulatedChannels.append(self.kernel.channel)
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
            copiedAccumulatedChannels = self.accumulatedChannels.copy()
            toReturn = [self.kernel.type, self.accumulatedX, self.accumulatedY, copiedAccumulatedChannels]
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
        freeDSPs = [dsp for dsp in self.DSPlist if not dsp.isActive()]
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
        pixelCounter = camera.pixelNumber()
        pixelCoords = camera.pixelCounterToXY(pixelCounter)
        pixelX = pixelCoords[0]
        pixelY = pixelCoords[1]
        
        if (targetY >= pixelY):
            raise Exception("ERROR! Camera pixel counter not advanced enough to do depthwise convolution")
        
        if (targetY + 1 == pixelY):
            if (targetX >= pixelX):
                raise Exception("ERROR! Camera pixel counter not advanced enough to do depthwise convolution")
        
        # Since layer 0 is where the camera output gets turned to 16 channels
        freeDSPs = self.getFreeDSPunits(channelsOut[0])
        
        self.assignOperationTag(freeDSPs, self.tagCounter)
        depthwiseConvOp = CameraDepthwiseConvOperation(tag=self.tagCounter, targetX=targetX, targetY=targetY)
        self.tagCounter += 1
        
        self.fifoPush(depthwiseConvOp)

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
        if isinstance(operation, CameraDepthwiseConvOperation):
            cameraActivation = imxCamera.getActivation(operation.targetX + operation.xOffset, operation.targetY + operation.yOffset)
            
            DSPunits = self.getDSPsWithOpTag(operation.tag)
            
            if (len(DSPunits) != 16):
                raise Exception("Camera Depthwise Conv Op attempted without 16 DSP units")
            
            channel = 0
            for DSPfromList in DSPunits:
                partFromOp = operation.xOffset + operation.yOffset * 3
                
                desiredWeight = DepthwiseWeight(layer=0, channel=channel, part=partFromOp)

                desiredActivation = Activation(type="depthwise", layer=0, x=cameraActivation.x, y=cameraActivation.y)
                
                # I actually think this is a bad way of doing this... Because, you have to do
                # a bunch of dumb stuff. Like if you did this in hardware, you have to have each
                # DSPunit have metadata about what activation and weight it has loaded, and then
                # the DSPmanager has to interrogate each DSPunit to make sure it has the right stuff...
                # Too much complexity, too much data
                
                # Instead, what I think should happen is each operation should have an array of bits.
                # We need to be able to query the memory manager and ask, if I ask you for this weight,
                # will you be able to get it to me in the next clock cycle?
                
                # Even better, we convert the weight into an LRAM, and then see if that LRAM is currently in
                # use this cycle. This is best because it parallels the way you do it in hardware. The
                # memory manager just sends back some LRAMUSED signals, and then the DSPmanager uses
                # those to determine which DSPs it can load with weights
                
                # But, the answer is even more complex than that, because, there are multiple weights packed per LRAM
                # read.
                # Low key, what if I make each depthwise convolution kernel only 8 weights. So it looks like
                # 1 1 1
                # 1 1 1
                # 1 1 0
                # That way, it fits cleanly into two LRAM reads, and makes life easier.
                # Ok, actually, I think it's better to have 9 activations per kernel for a really specific reason.
                # It's because there's only 5 LRAMs to begin with. So even if you could fit 1 kernel per 2 SRAMs,
                # you will still only be able to read 2 kernels at a time, since 3 reads would require
                # 2 * 3 = 6 simultaneous SRAM reads which is not possible. So either way, you are still limited to 2 kernel
                # reads per cycle.
                #
                # So actually, I think the best way for this to be is for the memory manager to send a signal signifying
                # that the amount of weights it has on its plate for this cycle has completely used up all LRAMs, OR
                # that the next kernel spans across a busy LRAM, so it's useless to even request another weight
                # of course, that LRAM will still be read, and its result will be cached by the memory manager.
                #
                # So then, the DSPmanager will be loading weights from the lowest channel to the highest, to match
                # what the LRAMs already have stored. Actually, since we know the kernels will be loaded sequentially, instead
                # of having an array of bits for each kernel, we can have an int which basically acts as a pointer to the
                # next available DSPUnit we can load. Then, we have a similar pointer for the activations coming out, so 
                # we can keep track of our progress on the operation on two fronts
                #
                # Since we can read a kernel in one cycle, I think it makes sense to have each DSP unit to have enough internal
                # storage to store 9 weights. But then one is wasted during pointwise convolution. What I can do, actually,
                # is round all channel numbers to be divisible by 9 instead of divisible by 8. That's a pretty easy solution. And
                # IMO it's better than making a kernel a non-square shape.
                #
                # We have 320 Kbits of distributed RAM, which I am assuming means flip flop storage. So if that's the case,
                # I think it makes sense to store a kernel per DSP unit since it only takes 2 KBits of Flip flop storage total
                #
                # The only thing, is that if we do decide to round up to 9, then one pointwise "mini-kernel" will take three reads
                # from SRAM. The thing is, there's no globally optimized solution. It's either that we waste one "mini-kernel"
                # byte per "exchange" with the DSPUnit, or we waste half of an SRAM read. I think wasting half of an SRAM read
                # is smarter.
                #
                # Ok, so I kind of lost the plot... what's the idea? ==> The idea is: store everything differently and match
                # the hardware instead of what I'm doing now which is dumb
                #
                # So I need to add a function to the memory manager called askForKernel(), and passes in a bunch of DSP
                # units. And then the memory manager promises that, on the next cycle, it will read from memory and feed
                # it's signal to all of the DSPUnits in the list. ONLY if it returns true. If it returns false, then
                # it basically says I cant do that my LRAMS are too busy. Also the memoryManager should cache its old
                # read if it reads one LRAM and it has parts of two different kernels that way we can use the same
                # LRAM next time we want to read
                #
                # I also need to add a function in memoryManager that allows it to "transform" a bunch of weight datatypes
                # into a kernel datatype. This has no meaning in hardware, but in software it just allows us to see more
                # easily about what we are doing.
                
                channel += 1

DSPMngr = DSPmanager()

NUM_LRAMS = 5
MAX_SIMULTANEOUS_LRAM_WRITES = NUM_LRAMS

WEIGHTS_PER_LRAM_ADDRESS = 4
class memoryManager:
    
    #SRAM will be used for activation, and LRAM for weight
    def __init__(self):
        
        self.allSRAM = []
        self.numSRAM = 166

        for idx in range(self.numSRAM):
            self.allSRAM.append(SRAM())

        self.allLRAM = []
        self.numLRAM = 5

        for idx in range(self.numLRAM):
            self.allLRAM.append(LRAM())
        
        # This is for loading the weights
        self.weightIndexCount = 0
        self.weightPack = []
        
        # This is for finding weights
        self.startOfLayer = np.zeros((layers, 3), dtype=int)
    
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
                if (LRAMobj.isActiveRead(currentAddr)):
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
            if (firstWeight.y % POINTWISE_KERNEL_PARTS != 0):
                raise Exception("ERROR! Pointwise kernel must start at a y divisible by 9")

            for idx, weight in enumerate(weightList):
                if (not isinstance(weight, PointwiseWeight)):
                    raise Exception("ERROR! Cannot make kernel from mixed weight types")
                if (weight.layer != firstWeight.layer or weight.channel != firstWeight.channel):
                    raise Exception("ERROR! Pointwise weights are not in the same kernel")
                if (weight.y != firstWeight.y + idx):
                    raise Exception("ERROR! Pointwise weights are not sequential")

            return PointwiseKernel(
                layer=firstWeight.layer,
                channel=firstWeight.channel,
                channelGroup=firstWeight.y // POINTWISE_KERNEL_PARTS)

        raise Exception("ERROR! Unsupported weight type for kernel conversion")

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
            
            kernel = self.weightsToKernel(weights)
            # Get DSPUnit by ID
            # Load kernel to DSPUnit
            
            dsp = DSPMngr.getDSPByID(id)
            
            dsp.preloadKernel(kernel)
        elif (isinstance(kernel, PointwiseKernel)):
            # I will implement this later, should be similar to code above, just need to add a check
            # for if it is at the part of the code where we store one half of a chunk near another
            return False
        else:
            raise Exception("Unrecognized Kernel type")
        
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
        kernelSize = DEPTHWISE_KERNEL_PARTS
        chunkSize = DEPTHWISE_KERNEL_PARTS
    elif (layerList[layer] == "pointwise"):
        kernelSize = channelsOut[layer]
        chunkSize = POINTWISE_KERNEL_PARTS
    else:
        raise Exception("ERROR! Unsupported convolution weight type")

    if (((NUM_DSPS//channelsOut[layer]) > (spatialSize[layer] // 4)) and layerList[layer] == "pointwise"):
        offset = 0
        totalParts = 0
        storeSecondHalf = False
        while offset*chunkSize < (kernelSize//2):
            for cycle in range(MAX_SIMULTANEOUS_LRAM_WRITES * WEIGHTS_PER_LRAM_ADDRESS):
                if (not storeSecondHalf):
                    totalParts = part+offset*chunkSize
                else:
                    totalParts = part+offset*chunkSize+(kernelSize//2)
                
                weight = PointwiseWeight(layer=layer, channel=channel,
                    y=totalParts)
                memory.loadWeightsInStart(weight)
                
                part += 1
                if (part >= chunkSize):
                    channel += 1
                    part = 0
                
                if (channel >= channelsIn[layer]):
                    channel = 0
                    if (storeSecondHalf):
                        offset += 1
                    storeSecondHalf = not storeSecondHalf
                
                if (offset*chunkSize >= (kernelSize//2)):
                    break
            clkCycles += 1
    else:
        offset = 0
        totalParts = 0
        while offset*chunkSize < kernelSize:
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
                
                if (offset*chunkSize >= kernelSize):
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

imxCamera = camera(INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE)

# Below is the behavior of the global controller:

# Before we start the image, we want to preload the 

imxCamera.startImage()

pixelsComputed = 0

while (pixelsComputed < INITIAL_IMAGE_SIZE ** 2):
    imxCamera.updateCamera()
    
    pixelNumber = imxCamera.pixelNumber()
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
        if (targetX + 1 <= pixelX):
            pass
            
    

# Until we have two rows of data, we can't do anything
for idx in range((128 + 2) * 30):
    imxCamera.updateCamera()

# Now we should have pixelNumber() == 128 + 2 = 130

print("Should be 130 (pixelNumber) : " + str(imxCamera.pixelNumber()))

