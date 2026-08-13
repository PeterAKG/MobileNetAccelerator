# This is the simulator to end all simulators... will rewrite from scratch. Want to support:
# Memory
# Squeeze and Excite
# And I want to make sure the order everything is being run in is actually intelligent

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from dataclasses import dataclass, field, replace, asdict, is_dataclass
from pathlib import Path
from typing import Literal, Optional, Union
from collections import deque
from typing import List

df = pd.read_excel('MobileNetV3Calc.xlsx')

NUM_MISC_LAYERS = 4
TRACKING_MODE_ON = True

layers = 39
layerList = np.zeros(layers, dtype="U20")
kernelSize = np.zeros(layers, dtype=int)
stride = np.zeros(layers, dtype=int)
channelsIn = np.zeros(layers, dtype=int)
channelsOut = np.zeros(layers, dtype=int)
spatialOutput = np.zeros(layers, dtype=int)
spatialInput = np.zeros(layers, dtype=int)
squeezeExcite = np.zeros(layers, dtype=int)
SEhiddenLayer = np.zeros(layers, dtype=int)
SRAMStartIndex = np.zeros(layers, dtype=int)
SEDSPsNeeded = np.zeros(layers, dtype=int)
SEMACsPerDSP = np.zeros(layers, dtype=int)
SEWEIGHTTOTALBYTES = int(df.loc[35, "SE Total NN Size (Bytes)"])
clkCycles = 0

clockHz = 300000000 # 300 MHZ Clock speed
convType = Literal["depthwise", "pointwise", "fullyConnected"]
ActivationType = Literal["depthwise", "pointwise"]
SEVectorType = Literal["input", "output"]

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
        spatialOutput[idx] = int(row["Spatial Output"])
        spatialInput[idx] = int(row["Spatial Input"])
        if(row["SE"] == "Yes"):
            squeezeExcite[idx] = 1
            SEhiddenLayer[idx] = int(row["SE Hidden Layer"])
            SEMACsPerDSP[idx] = int(row["SE Squeeze MACs / DSP"])
            SEDSPsNeeded[idx] = int(row["Squeeze DSPs Needed"])
        else:
            squeezeExcite[idx] = 0
        SRAMStartIndex[idx] = int(row["SRAM Activation Start Index"])


KERNEL_WEIGHT_CAP = 9
DEPTHWISE_KERNEL_PARTS = KERNEL_WEIGHT_CAP
POINTWISE_KERNEL_PARTS = KERNEL_WEIGHT_CAP
SE_VECTOR_CHUNK_PARTS = 9

def makeJSONSerializable(value):
    if (value is None):
        return None
    if (is_dataclass(value)):
        return makeJSONSerializable(asdict(value))
    if (isinstance(value, dict)):
        return {key: makeJSONSerializable(val) for key, val in value.items()}
    if (isinstance(value, (list, tuple))):
        return [makeJSONSerializable(item) for item in value]
    if (isinstance(value, np.integer)):
        return int(value)
    if (isinstance(value, np.floating)):
        return float(value)
    if (isinstance(value, np.ndarray)):
        return makeJSONSerializable(value.tolist())
    return value

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
class SEKernel:
    type: Literal["SE"] = field(default="SE", init=False)
    weights: List[SEWeight] = field(default_factory=list)

    def __post_init__(self):
        if (len(self.weights) > KERNEL_WEIGHT_CAP):
            raise Exception(f"ERROR! SE kernel cannot contain more than {KERNEL_WEIGHT_CAP} weights")

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
    PointwiseKernel,
    SEKernel
]

# Channel argument refers to the input channel
@dataclass
class Activation:
    type: ActivationType
    layer: int
    x: int
    y: int
    channel: int
    value: np.uint8 = np.uint8(1)
    
@dataclass
class ActivationChunk:
    type: ActivationType
    layer: int = -1
    x: int = -1
    y: int = -1
    channel: int = -1
    channelChunk: int = -1   # For pointwise it represents the lowest channel.

@dataclass
class SEVectorIndex:
    type: SEVectorType
    layer: int = -1
    channel: int = -1
    value: np.uint8 = np.uint8(0)

@dataclass
class SEVectorChunk:
    type: SEVectorType
    layer: int = -1
    channel: int = -1   # represents the lowest channel of the group of 9
    valuesStored: int = 0

@dataclass
class SEhiddenLayerElem:
    elemNumber: int

@dataclass
class SEHiddenLayerVector:
    length: int

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
    channelsCompleted: int = 0

@dataclass
class SEVectorProdOperation:
    layer: int
    tag: int
    weightsPerDSP: int
    squeezeOperationComplete: bool = False
    exciteOperationStarted: bool = False
    exciteOperationComplete: bool = False
    exciteTag: Optional[int] = None
    isExpansion: bool = False
    weightsLoaded: int = 0
    DSPindexStart: int = 0
    currentLoadingIndex: int = 0
    currentUnloadingIndex: int = 0
    currentOutputStoreIndex: int = 0
    accumulatorRowsLinked: bool = False
    extractedOutputs: List[Union[SEhiddenLayerElem, SEVectorIndex]] = field(default_factory=list)

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
    SEVectorProdOperation,
    FullyConnectedOperation,
]

@dataclass
class LoadSEWeightsMemOp:
    DSPtag: int
    memTag: int
    isExpansion: bool
    layer: int
    weightsPerDSP: int
    weightsLoaded: int
    DSPindexStart: int = 0

MemoryOperation = Union[
    LoadSEWeightsMemOp
]

class SRAM:
    
    def __init__(self):
        self.storage = 18432 # bits
        self.portSize = 18 # bits
        self.addresses = SRAM_ADDRESSES # Number of available addresses
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

    def exportAsJSON(self):
        addresses = []
        for address in range(self.addresses):
            addresses.append({
                "address": address,
                "data": makeJSONSerializable(list(self.memory[address]))
            })
        return {
            "addresses": addresses
        }
        
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

    def exportAsJSON(self):
        addresses = []
        for address in range(self.addresses):
            addresses.append({
                "address": address,
                "data": makeJSONSerializable(list(self.memory[address]))
            })
        return {
            "addresses": addresses
        }

# I don't think I am going to use this and I'm not sure why I ever planned to
# Or maybe I will. I honestly don't know
# Honestly -- probably not. It doesn't seem like it would save many cycles, and it increases complexity
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

PIXEL_HISTORY_MAP_DEPTH = 3
PIXEL_HISTORY_MAP_SIZE = 64
# This tracks the OUTPUT of each respective layer. Not the INPUT.
class pixelHistorian:
    
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.pixelHistoryMap = np.zeros((width, height, depth), dtype=bool)
        self.SEVectorCalculated = np.zeros(depth, dtype=bool)
        self.layerForSlot = np.arange(depth, dtype=int)
        self.lastCompletedLayer = -1
        self.layerPointer = 0

    def getLayerSlot(self, layer):
        if (layer < 0):
            raise Exception("ERROR! Pixel layer cannot be negative")

        matches = np.where(self.layerForSlot == layer)[0]
        if (len(matches) == 0):
            minTrackedLayer = int(np.min(self.layerForSlot))
            maxTrackedLayer = int(np.max(self.layerForSlot))
            raise Exception(f"ERROR! Pixel layer {layer} is outside the tracked layer window ({minTrackedLayer}-{maxTrackedLayer})")

        return int(matches[0])
    
    def setPixelFinished(self, x, y, layer):
        if (x < 0 or x >= self.width):
            raise Exception("ERROR! Pixel x coordinate is outside the pixel history map width")
        if (y < 0 or y >= self.height):
            raise Exception("ERROR! Pixel y coordinate is outside the pixel history map height")

        layerSlot = self.getLayerSlot(layer)
        layerSpatialSize = spatialOutput[layer]
        if (x >= layerSpatialSize or y >= layerSpatialSize):
            raise Exception(f"ERROR! Pixel ({x}, {y}) is outside layer {layer}'s spatial output ({layerSpatialSize}x{layerSpatialSize})")

        self.pixelHistoryMap[x, y, layerSlot] = True

        completedLayer = int(self.layerForSlot[self.layerPointer])
        completedSpatialSize = spatialOutput[completedLayer]
        while (self.pixelHistoryMap[:completedSpatialSize, :completedSpatialSize, self.layerPointer].all()):
            if (completedLayer != self.lastCompletedLayer + 1):
                break

            self.lastCompletedLayer = completedLayer
            nextLayer = self.lastCompletedLayer + self.depth
            self.pixelHistoryMap[:, :, self.layerPointer] = False
            self.SEVectorCalculated[self.layerPointer] = False
            self.layerForSlot[self.layerPointer] = nextLayer
            self.layerPointer = (self.layerPointer + 1) % self.depth
            completedLayer = int(self.layerForSlot[self.layerPointer])
            completedSpatialSize = spatialOutput[completedLayer]

    def isQuarterFinished(self, layer, quarter):
        layerSlot = self.getLayerSlot(layer)
        layerSpatialSize = spatialOutput[layer]
        halfSpatialSize = layerSpatialSize // 2

        if (quarter == 1):
            xStart = 0
            xEnd = halfSpatialSize + 1
            yStart = 0
            yEnd = halfSpatialSize + 1
        elif (quarter == 2):
            xStart = halfSpatialSize - 1
            xEnd = layerSpatialSize
            yStart = 0
            yEnd = halfSpatialSize + 1
        elif (quarter == 3):
            xStart = 0
            xEnd = halfSpatialSize + 1
            yStart = halfSpatialSize - 1
            yEnd = layerSpatialSize
        elif (quarter == 4):
            xStart = halfSpatialSize - 1
            xEnd = layerSpatialSize
            yStart = halfSpatialSize - 1
            yEnd = layerSpatialSize
        else:
            raise Exception("ERROR! Quarter must be 1, 2, 3, or 4")

        return bool(self.pixelHistoryMap[xStart:xEnd, yStart:yEnd, layerSlot].all())

    def isLayerFinished(self, layer):
        if (layer < 0):
            raise Exception("ERROR! Pixel layer cannot be negative")
        return layer <= self.lastCompletedLayer

    def setSEVectorCalculated(self, layer, calculated=True):
        layerSlot = self.getLayerSlot(layer)
        self.SEVectorCalculated[layerSlot] = calculated

    def isSEVectorCalculated(self, layer):
        layerSlot = self.getLayerSlot(layer)
        return bool(self.SEVectorCalculated[layerSlot])
    
    def getLastCompletedLayer(self):
        return self.lastCompletedLayer

pixelMap = pixelHistorian(PIXEL_HISTORY_MAP_SIZE, PIXEL_HISTORY_MAP_SIZE, PIXEL_HISTORY_MAP_DEPTH)

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

class DSPAccumulator:
    def __init__(self, accumulatorID):
        self.accumulatorID = accumulatorID
        self.DSPunits = []
        self.linkedAccumulators = []
        self.linkedOutputReadyClock = None
        self.accumulatedDSPunits = []
        self.accumulatedChannels = []
        self.accumulatedType = None
        self.accumulatedLayer = -1
        self.accumulatedOutputChannel = -1
        self.accumulatedX = -1
        self.accumulatedY = -1

    def addDSPunit(self, DSPunit):
        self.DSPunits.append(DSPunit)
        DSPunit.accumulator = self

    def isFree(self):
        return all((not dsp.isActive()) and dsp.getOperationTag() is None for dsp in self.DSPunits)

    def isActive(self):
        return any(dsp.isActive() for dsp in self.DSPunits)

    def linkedAccumulatorCount(self):
        return len(self.getLinkedAccumulatorChain())

    def getLinkedAccumulatorChain(self):
        linkedAccumulatorChain = []
        seenAccumulators = set()

        def collectLinkedAccumulators(accumulator):
            for linkedAccumulator in accumulator.linkedAccumulators:
                if (linkedAccumulator in seenAccumulators):
                    continue
                seenAccumulators.add(linkedAccumulator)
                linkedAccumulatorChain.append(linkedAccumulator)
                collectLinkedAccumulators(linkedAccumulator)

        collectLinkedAccumulators(self)
        return linkedAccumulatorChain

    def linkAccumulator(self, accumulator):
        if (not isinstance(accumulator, DSPAccumulator)):
            raise Exception("ERROR! Can only link a DSPAccumulator to another DSPAccumulator")
        if (accumulator is self):
            raise Exception("ERROR! DSP accumulator cannot link to itself")
        if (accumulator in self.linkedAccumulators):
            return
        if (self in accumulator.getLinkedAccumulatorChain()):
            raise Exception("ERROR! DSP accumulator links cannot contain cycles")

        self.linkedAccumulators.append(accumulator)
        self.linkedOutputReadyClock = None

    def unlinkAccumulator(self, accumulator):
        if (accumulator not in self.linkedAccumulators):
            return

        self.linkedAccumulators.remove(accumulator)
        self.linkedOutputReadyClock = None

    def ownHasOutput(self):
        outputDSPunits = self.outputDSPunits()
        if (len(outputDSPunits) == 0):
            return False
        return all(self.DSPHasOutputMetadata(dsp) for dsp in outputDSPunits)

    def ownHasOutputReady(self):
        return self.ownHasOutput() and (not self.isActive())

    def allLinkedAccumulatorsHaveOutputReady(self):
        return self.ownHasOutputReady() and all(
            accumulator.ownHasOutputReady()
            for accumulator in self.getLinkedAccumulatorChain())

    def refreshLinkedOutputReadyClock(self):
        if (not self.allLinkedAccumulatorsHaveOutputReady()):
            self.linkedOutputReadyClock = None
            return

        if (self.linkedOutputReadyClock is None):
            self.linkedOutputReadyClock = clkCycles

    def DSPHasOutputMetadata(self, DSPunit):
        if (isinstance(DSPunit.kernel, SEKernel)):
            return self.accumulatedType in ("SE", "SEExpansion") and DSPunit in self.accumulatedDSPunits
        return (
            DSPunit.kernel.layer >= 0 and
            DSPunit.activationChunk.layer >= 0 and
            DSPunit.kernel.type == DSPunit.activationChunk.type)

    def outputDSPunits(self):
        if (len(self.accumulatedDSPunits) > 0):
            return self.accumulatedDSPunits
        return self.DSPunits

    def hasOutput(self):
        return self.ownHasOutput() and all(
            accumulator.ownHasOutput()
            for accumulator in self.getLinkedAccumulatorChain())

    def hasOutputReady(self):
        self.refreshLinkedOutputReadyClock()
        if (self.linkedOutputReadyClock is None):
            return False
        return (clkCycles - self.linkedOutputReadyClock) >= self.linkedAccumulatorCount()

    def outputMetadataForDSP(self, DSPunit):
        if (isinstance(DSPunit.kernel, SEKernel)):
            accumulator = DSPunit.accumulator
            return (
                accumulator.accumulatedType,
                accumulator.accumulatedLayer,
                accumulator.accumulatedOutputChannel,
                accumulator.accumulatedX,
                accumulator.accumulatedY)

        return (
            DSPunit.kernel.type,
            DSPunit.kernel.layer,
            DSPunit.kernel.channel,
            DSPunit.activationChunk.x,
            DSPunit.activationChunk.y)

    def validateMatchingOutputMetadata(self, DSPunit):
        accumulatedMetadata = (
            self.accumulatedType,
            self.accumulatedLayer,
            self.accumulatedOutputChannel,
            self.accumulatedX,
            self.accumulatedY)
        dspMetadata = self.outputMetadataForDSP(DSPunit)
        if (accumulatedMetadata != dspMetadata):
            raise Exception("ERROR! DSP accumulator received mismatched output metadata")

    def recordDotProductStart(self, DSPunit):
        if (DSPunit not in self.DSPunits):
            raise Exception("ERROR! DSP attempted to record output with an accumulator it is not wired to")

        if (self.accumulatedType is None):
            (
                self.accumulatedType,
                self.accumulatedLayer,
                self.accumulatedOutputChannel,
                self.accumulatedX,
                self.accumulatedY) = self.outputMetadataForDSP(DSPunit)
        else:
            self.validateMatchingOutputMetadata(DSPunit)

        if (DSPunit not in self.accumulatedDSPunits):
            self.accumulatedDSPunits.append(DSPunit)

        if(DSPunit.kernel.type == "pointwise"):
            self.accumulatedChannels.append(DSPunit.kernel.channelGroup)

    def recordSEDotProductStart(self, DSPunit, layer, row, isExpansion=False):
        if (DSPunit not in self.DSPunits):
            raise Exception("ERROR! DSP attempted to record SE output with an accumulator it is not wired to")
        if (not isinstance(DSPunit.kernel, SEKernel)):
            raise Exception("ERROR! DSP accumulator attempted to record SE output from a non-SE kernel")

        accumulatedMetadata = (
            self.accumulatedType,
            self.accumulatedLayer,
            self.accumulatedOutputChannel,
            self.accumulatedX,
            self.accumulatedY)
        dspMetadata = ("SEExpansion" if isExpansion else "SE", layer, row, -1, -1)

        if (self.accumulatedType is None):
            (
                self.accumulatedType,
                self.accumulatedLayer,
                self.accumulatedOutputChannel,
                self.accumulatedX,
                self.accumulatedY) = dspMetadata
        elif (accumulatedMetadata != dspMetadata):
            raise Exception("ERROR! DSP accumulator received mismatched SE output metadata")

        if (DSPunit not in self.accumulatedDSPunits):
            self.accumulatedDSPunits.append(DSPunit)

    def clearAccumulation(self):
        self.accumulatedDSPunits.clear()
        self.accumulatedChannels.clear()
        self.accumulatedType = None
        self.accumulatedLayer = -1
        self.accumulatedOutputChannel = -1
        self.accumulatedX = -1
        self.accumulatedY = -1
        self.linkedOutputReadyClock = None

    def getOutput(self):
        if (not self.hasOutput()):
            raise Exception("ERROR! Activation was attempted to be taken from empty DSP accumulator")

        if (not self.allLinkedAccumulatorsHaveOutputReady()):
            raise Exception("ERROR! Activation was attempted to be taken from active DSP accumulator")

        if (not self.hasOutputReady()):
            raise Exception("ERROR! Activation was attempted to be taken from DSP accumulator before linked accumulator delay elapsed")

        accumulatorsToRead = [self] + self.getLinkedAccumulatorChain()

        for accumulator in accumulatorsToRead:
            for DSPunit in accumulator.outputDSPunits():
                self.validateMatchingOutputMetadata(DSPunit)

        outputStride = stride[self.accumulatedLayer]
        if (self.accumulatedType == "SE"):
            toReturn = SEhiddenLayerElem(elemNumber=self.accumulatedOutputChannel)
        elif (self.accumulatedType == "SEExpansion"):
            toReturn = SEVectorIndex(
                type="output",
                layer=self.accumulatedLayer,
                channel=self.accumulatedOutputChannel)
        elif (self.accumulatedType == "depthwise"):
            toReturn = Activation(
                type="depthwise",
                layer=self.accumulatedLayer,
                x=self.accumulatedX // outputStride,
                y=self.accumulatedY // outputStride,
                channel=self.accumulatedOutputChannel)
        else:
            accumulatedChannelSet = set()
            for accumulator in accumulatorsToRead:
                accumulatedChannelSet.update(accumulator.accumulatedChannels)
            usedAllChannels = len(accumulatedChannelSet) == channelsIn[self.accumulatedLayer]

            if (usedAllChannels):
                toReturn = Activation(
                    type="pointwise",
                    layer=self.accumulatedLayer,
                    x=self.accumulatedX // outputStride,
                    y=self.accumulatedY // outputStride,
                    channel=self.accumulatedOutputChannel)
            else:
                toReturn = PartialActivation(
                    layers=[self.accumulatedLayer],
                    x=self.accumulatedX // outputStride,
                    y=self.accumulatedY // outputStride,
                    segment=min(accumulatedChannelSet),
                    outputChannel=self.accumulatedOutputChannel)

        return toReturn

    def clearOutput(self):
        if (not self.hasOutput()):
            raise Exception("ERROR! Attempted to clear an empty DSP accumulator")
        if (not self.allLinkedAccumulatorsHaveOutputReady()):
            raise Exception("ERROR! Attempted to clear an active DSP accumulator")

        accumulatorsToClear = [self] + self.getLinkedAccumulatorChain()

        for accumulator in accumulatorsToClear:
            for DSPunit in accumulator.outputDSPunits():
                DSPunit.clearActiveStorage()

        for accumulator in accumulatorsToClear:
            accumulator.clearAccumulation()

class DSPunit:
    def __init__(self, dspID):
        self.dspID = dspID

        self.kernel = DepthwiseKernel()
        
        self.preloadedKernel = DepthwiseKernel()
        
        self.activationChunk = ActivationChunk(type="Depthwise")
        
        self.preloadedActivationChunk = ActivationChunk(type="Depthwise")

        self.preAdder = SEVectorChunk(type="input")

        self.preloadedPreAdder = SEVectorChunk(type="input")
        
        # Amount of time it takes until it has multiplied all together. When self.timer == 0, it is finished
        self.timer = -1
        self.lastTimerUpdateClockCycle = clkCycles
        
        self.accumulator = None
        
        # Operation tag -- used for the DSPmanager to mark which operations correspond to which DSPunits
        self.operationTag = None
        
    def preloadKernel(self, kernel: Kernel):
        if (isinstance(kernel, SEKernel)):
            self.preloadedKernel = SEKernel(weights=kernel.weights.copy())
        else:
            self.preloadedKernel = kernel
    
    def loadKernel(self):
        if (isinstance(self.preloadedKernel, SEKernel)):
            self.kernel = SEKernel(weights=self.preloadedKernel.weights.copy())
        else:
            self.kernel = replace(self.preloadedKernel)
        self.preloadedKernel = DepthwiseKernel()
    
    def preloadActivation(self, activationChunk: ActivationChunk):
        self.preloadedActivationChunk = activationChunk
        
    def loadActivation(self):
        self.activationChunk = replace(self.preloadedActivationChunk)
        self.preloadedActivationChunk = ActivationChunk(type=self.activationChunk.type)

    def preloadPreAdder(self, seVectorChunk: SEVectorChunk):
        if (not isinstance(seVectorChunk, SEVectorChunk)):
            raise Exception("ERROR! DSP pre-adder must be preloaded with an SEVectorChunk")

        self.preloadedPreAdder = replace(seVectorChunk)

    def loadPreAdder(self):
        self.preAdder = replace(self.preloadedPreAdder)
        self.preloadedPreAdder = SEVectorChunk(type=self.preAdder.type)

    def clearActiveStorage(self):
        self.kernel = DepthwiseKernel()
        self.activationChunk = ActivationChunk(type="Depthwise")
        self.preAdder = SEVectorChunk(type="input")

    def hasPreloadedKernel(self):
        if (isinstance(self.preloadedKernel, SEKernel)):
            return len(self.preloadedKernel.weights) > 0
        return self.preloadedKernel.layer >= 0

    def hasPreloadedActivation(self):
        return self.preloadedActivationChunk.layer >= 0

    def hasPreloadedPreAdder(self):
        return isinstance(self.preloadedPreAdder, SEVectorChunk) and self.preloadedPreAdder.valuesStored > 0

    def hasPreloadedDotProduct(self):
        if (not self.hasPreloadedKernel()):
            return False
        if (isinstance(self.preloadedKernel, SEKernel)):
            return self.hasPreloadedPreAdder()
        if (not self.hasPreloadedActivation()):
            return False
        if (squeezeExcite[self.preloadedActivationChunk.layer] == 1):
            return self.hasPreloadedPreAdder()
        return True

    def validatePreAdderForActivation(self):
        if (squeezeExcite[self.activationChunk.layer] != 1):
            return

        if (not isinstance(self.preAdder, SEVectorChunk)):
            raise Exception("ERROR! Squeeze-excite activation started without a loaded DSP pre-adder SE vector chunk")
        if (self.preAdder.layer != self.activationChunk.layer):
            raise Exception("ERROR! Squeeze-excite DSP pre-adder layer does not match activation chunk layer")

        expectedChannel = (self.activationChunk.channel // SE_VECTOR_CHUNK_PARTS) * SE_VECTOR_CHUNK_PARTS
        if (self.preAdder.channel != expectedChannel):
            raise Exception("ERROR! Squeeze-excite DSP pre-adder channel does not match activation chunk channel")
        
    def startDotProductUnchecked(self):
        if (isinstance(self.kernel, SEKernel)):
            if (not isinstance(self.preAdder, SEVectorChunk) or self.preAdder.valuesStored <= 0):
                raise Exception("ERROR! SE dot product started without a loaded SE vector chunk")
            if (len(self.kernel.weights) != self.preAdder.valuesStored):
                raise Exception("ERROR! SE dot product weight count does not match SE vector chunk count")
            self.timer = 8
            self.lastTimerUpdateClockCycle = clkCycles
            return

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
        else:
            self.validatePreAdderForActivation()
            self.timer = 8
            self.lastTimerUpdateClockCycle = clkCycles
            
            if (self.accumulator is None):
                raise Exception("ERROR! DSP slice started without an assigned accumulator")
            self.accumulator.recordDotProductStart(self)

    def startDotProduct(self):
        if (self.isActive()):
            raise Exception("ERROR! DSP Slice attempted to start while already being active!")
        self.startDotProductUnchecked()

    def startPreloadedDotProductIfAvailable(self):
        if (not self.hasPreloadedDotProduct()):
            return False

        preloadedKernelIsSE = isinstance(self.preloadedKernel, SEKernel)
        self.loadKernel()
        if (preloadedKernelIsSE):
            self.loadPreAdder()
        else:
            self.loadActivation()
        if ((not preloadedKernelIsSE) and squeezeExcite[self.activationChunk.layer] == 1):
            self.loadPreAdder()
        self.startDotProductUnchecked()
        return True
    
    def updateTimer(self):
        if(self.timer > 0):
            elapsedClockCycles = max(0, clkCycles - self.lastTimerUpdateClockCycle)
            self.timer = max(0, self.timer - elapsedClockCycles)
            self.lastTimerUpdateClockCycle = clkCycles
            if (self.timer == 0):
                self.startPreloadedDotProductIfAvailable()
    
    def isActive(self):
        self.updateTimer()
        return (self.timer > 0)
    
    def getOutput(self):
        raise Exception("ERROR! DSP output must be read from its assigned accumulator")
    
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
DSPS_PER_ACCUMULATOR = 4
class DSPmanager:
    def __init__(self):
        self.DSPlist = []
        self.accumulators = []
        if (NUM_DSPS % DSPS_PER_ACCUMULATOR != 0):
            raise Exception("ERROR! NUM_DSPS must be divisible by DSPS_PER_ACCUMULATOR")

        for accumulatorID in range(NUM_DSPS // DSPS_PER_ACCUMULATOR):
            self.accumulators.append(DSPAccumulator(accumulatorID))

        for i in range(NUM_DSPS):
            newDSP = DSPunit(i)
            self.accumulators[i // DSPS_PER_ACCUMULATOR].addDSPunit(newDSP)

            # A dsp unit and a number indicating whether it has been used this cycle
            self.DSPlist.append(newDSP)
        
        self.operationFIFO = deque()
        
        self.tagCounter = 0
        self.DSPActivationFirstClock = None
        self.DSPActivationLastClock = None
        self.DSPActivationCycleCount = 0
        self.DSPActiveRuns = [[] for _ in range(NUM_DSPS)]
        self.DSPOpenRunStarts = [None] * NUM_DSPS
    
    def getFreeDSPunits(self, max=None):
        freeDSPs = [dsp for dsp in self.DSPlist if (not dsp.isActive() and dsp.getOperationTag() is None)]
        if (max is None):
            return freeDSPs
        if (max < 0):
            raise Exception("ERROR! Maximum number of free DSP units cannot be negative")
        return freeDSPs[:max]

    def getFreeDSPsFromDistinctAccumulators(self, max=None):
        if (max is not None and max < 0):
            raise Exception("ERROR! Maximum number of free DSP units cannot be negative")

        freeDSPs = []
        for accumulator in self.accumulators:
            if (not accumulator.isFree()):
                continue
            for dsp in accumulator.DSPunits:
                freeDSPs.append(dsp)
                break

            if (max is not None and len(freeDSPs) >= max):
                break

        return freeDSPs

    def getFreeAccumulatorDSPGroups(self, max=None):
        if (max is not None and max < 0):
            raise Exception("ERROR! Maximum number of free DSP accumulator groups cannot be negative")

        freeDSPGroups = []
        for accumulator in self.accumulators:
            if (not accumulator.isFree()):
                continue
            freeDSPGroups.append(accumulator.DSPunits)
            if (max is not None and len(freeDSPGroups) >= max):
                break

        return freeDSPGroups

    def linkAccumulator(self, finalAccumulator, linkedAccumulator):
        finalAccumulator.linkAccumulator(linkedAccumulator)

    def unlinkAccumulator(self, finalAccumulator, linkedAccumulator):
        finalAccumulator.unlinkAccumulator(linkedAccumulator)

    def getSEAccumulatorRows(self, operation: SEVectorProdOperation, DSPunits):
        DSPsPerRow = self.getSEDSPsPerRow(
            operation.layer,
            operation.weightsPerDSP,
            operation.isExpansion)
        outputVectorLength = self.getSEOutputVectorLength(
            operation.layer,
            operation.isExpansion)

        return [
            DSPunits[rowStart:rowStart + DSPsPerRow]
            for rowStart in range(0, DSPsPerRow * outputVectorLength, DSPsPerRow)
        ]

    def linkSEAccumulatorRows(self, operation: SEVectorProdOperation, DSPunits):
        for rowDSPs in self.getSEAccumulatorRows(operation, DSPunits):
            if (len(rowDSPs) == 0):
                continue

            finalAccumulator = rowDSPs[0].accumulator
            linkedAccumulators = set()
            for DSPunit in rowDSPs[1:]:
                if (DSPunit.accumulator is finalAccumulator or DSPunit.accumulator in linkedAccumulators):
                    continue
                finalAccumulator.linkAccumulator(DSPunit.accumulator)
                linkedAccumulators.add(DSPunit.accumulator)

        operation.accumulatorRowsLinked = True

    def unlinkSEAccumulatorRow(self, rowDSPs):
        if (len(rowDSPs) == 0):
            return

        finalAccumulator = rowDSPs[0].accumulator
        linkedAccumulators = set()
        for DSPunit in rowDSPs[1:]:
            if (DSPunit.accumulator is finalAccumulator or DSPunit.accumulator in linkedAccumulators):
                continue
            finalAccumulator.unlinkAccumulator(DSPunit.accumulator)
            linkedAccumulators.add(DSPunit.accumulator)

    def recordSEAccumulatorStart(self, DSPunit, operation: SEVectorProdOperation, dspIndex):
        DSPsPerRow = self.getSEDSPsPerRow(
            operation.layer,
            operation.weightsPerDSP,
            operation.isExpansion)
        row = dspIndex // DSPsPerRow
        DSPunit.accumulator.recordSEDotProductStart(
            DSPunit,
            operation.layer,
            row,
            operation.isExpansion)

    def extractSEVectorOutputs(self, operation: SEVectorProdOperation, DSPunits):
        rowDSPsList = self.getSEAccumulatorRows(operation, DSPunits)

        while (operation.currentUnloadingIndex < len(rowDSPsList)):
            rowDSPs = rowDSPsList[operation.currentUnloadingIndex]
            finalAccumulator = rowDSPs[0].accumulator

            if (not finalAccumulator.hasOutputReady()):
                break

            operation.extractedOutputs.append(finalAccumulator.getOutput())
            finalAccumulator.clearOutput()
            self.unlinkSEAccumulatorRow(rowDSPs)
            operation.currentUnloadingIndex += 1

        if (operation.currentUnloadingIndex < len(rowDSPsList)):
            return False

        if (operation.isExpansion):
            while (operation.currentOutputStoreIndex < len(operation.extractedOutputs)):
                SEVectorIndexObj = operation.extractedOutputs[operation.currentOutputStoreIndex]
                if (memory.loadSEVectorOutput(SEVectorIndexObj) == False):
                    return False
                operation.currentOutputStoreIndex += 1
            return True

        memory.SEhiddenLayerElemsToSEHiddenLayerVector(operation.extractedOutputs)
        return True

    def getSEInputVectorLength(self, layer, isExpansion=False):
        if (isExpansion):
            return SEhiddenLayer[layer]
        return channelsOut[layer]

    def getSEOutputVectorLength(self, layer, isExpansion=False):
        if (isExpansion):
            return channelsOut[layer]
        return SEhiddenLayer[layer]

    def getSEDSPsPerRow(self, layer, weightsPerDSP, isExpansion=False):
        if (weightsPerDSP <= 0):
            raise Exception("ERROR! SE MACs per DSP must be positive")
        inputVectorLength = self.getSEInputVectorLength(layer, isExpansion)
        return math.ceil(inputVectorLength / weightsPerDSP)

    def getSEDSPsNeededForMatrix(self, layer, weightsPerDSP, isExpansion=False):
        return self.getSEDSPsPerRow(layer, weightsPerDSP, isExpansion) * self.getSEOutputVectorLength(layer, isExpansion)

    def getSEDSPSpanForRows(self, layer, weightsPerDSP, isExpansion=False):
        DSPsPerRow = self.getSEDSPsPerRow(layer, weightsPerDSP, isExpansion)
        outputVectorLength = self.getSEOutputVectorLength(layer, isExpansion)
        nextDSPIndex = 0

        for row in range(outputVectorLength):
            rowDSPs = []
            rowAccumulators = set()

            while (nextDSPIndex < len(self.DSPlist) and len(rowDSPs) < DSPsPerRow):
                DSPunit = self.DSPlist[nextDSPIndex]

                rowDSPs.append(DSPunit)
                rowAccumulators.add(DSPunit.accumulator)
                nextDSPIndex += 1

            if (len(rowDSPs) != DSPsPerRow):
                return -1

            while (nextDSPIndex < len(self.DSPlist) and self.DSPlist[nextDSPIndex].accumulator in rowAccumulators):
                nextDSPIndex += 1

        return nextDSPIndex

    def getFreeDSPsForSERows(self, layer, weightsPerDSP, isExpansion=False):
        DSPsPerRow = self.getSEDSPsPerRow(layer, weightsPerDSP, isExpansion)
        outputVectorLength = self.getSEOutputVectorLength(layer, isExpansion)
        freeDSPs = []
        nextDSPIndex = 0

        for row in range(outputVectorLength):
            rowDSPs = []
            rowAccumulators = set()

            while (nextDSPIndex < len(self.DSPlist) and len(rowDSPs) < DSPsPerRow):
                DSPunit = self.DSPlist[nextDSPIndex]
                if (DSPunit.isActive() or DSPunit.getOperationTag() is not None):
                    return []

                rowDSPs.append(DSPunit)
                rowAccumulators.add(DSPunit.accumulator)
                nextDSPIndex += 1

            if (len(rowDSPs) != DSPsPerRow):
                return []

            freeDSPs.extend(rowDSPs)

            while (nextDSPIndex < len(self.DSPlist) and self.DSPlist[nextDSPIndex].accumulator in rowAccumulators):
                DSPunit = self.DSPlist[nextDSPIndex]
                if (DSPunit.isActive() or DSPunit.getOperationTag() is not None):
                    return []
                nextDSPIndex += 1

        return freeDSPs
    
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
    
    def fifoPeek(self, distanceFromTop) -> Operation:
        index = -(distanceFromTop + 1)
        return self.operationFIFO[index]

    def fifoRemove(self, distanceFromTop) -> Operation:
        index = -(distanceFromTop + 1)
        operation = self.operationFIFO[index]
        del self.operationFIFO[index]
        return operation
    
    def fifoCheckLen(self) -> int:
        return len(self.operationFIFO)

    def isSEVectorCalculating(self):
        return any(isinstance(operation, SEVectorProdOperation) for operation in self.operationFIFO)

    def processSESqueezeOperation(self, operation: SEVectorProdOperation):
        DSPunits = self.getDSPsWithOpTag(operation.tag)
        expectedDSPCount = self.getSEDSPsNeededForMatrix(
            operation.layer,
            operation.weightsPerDSP,
            operation.isExpansion)
        if (len(DSPunits) != expectedDSPCount):
            raise Exception("ERROR! SEVectorProdOperation must load all DSP units assigned to the SE operation")
        if (operation.weightsPerDSP < 0):
            raise Exception("ERROR! SEVectorProdOperation weightsPerDSP cannot be negative")
        if (operation.weightsLoaded < 0):
            raise Exception("ERROR! SEVectorProdOperation weightsLoaded cannot be negative")
        if (operation.weightsLoaded > operation.weightsPerDSP):
            raise Exception("ERROR! SEVectorProdOperation weightsLoaded cannot exceed weightsPerDSP")
        if (operation.DSPindexStart < 0 or operation.DSPindexStart > len(DSPunits)):
            raise Exception("ERROR! SEVectorProdOperation DSPindexStart is outside the DSP range")

        if (not operation.accumulatorRowsLinked):
            self.linkSEAccumulatorRows(operation, DSPunits)

        remainingWeightsPerDSP = operation.weightsPerDSP - operation.weightsLoaded
        if (remainingWeightsPerDSP == 0):
            return self.extractSEVectorOutputs(operation, DSPunits)

        for dspIndex in range(operation.DSPindexStart, len(DSPunits)):
            DSPunit = DSPunits[dspIndex]
            DSPunit.isActive()
            hasSEKernelLoaded = isinstance(DSPunit.kernel, SEKernel)
            hasSEKernelPreloaded = isinstance(DSPunit.preloadedKernel, SEKernel)
            hasSEPreAdderLoaded = isinstance(DSPunit.preAdder, SEVectorChunk) and DSPunit.preAdder.valuesStored > 0
            hasSEPreAdderPreloaded = isinstance(DSPunit.preloadedPreAdder, SEVectorChunk) and DSPunit.preloadedPreAdder.valuesStored > 0

            if ((not DSPunit.isActive()) and hasSEKernelLoaded and (not hasSEKernelPreloaded)):
                DSPunit.clearActiveStorage()
                hasSEKernelLoaded = False
                hasSEPreAdderLoaded = False

            if ((not DSPunit.isActive()) and DSPunit.hasPreloadedDotProduct()):
                DSPunit.startPreloadedDotProductIfAvailable()
                self.recordSEAccumulatorStart(DSPunit, operation, dspIndex)
                hasSEKernelLoaded = isinstance(DSPunit.kernel, SEKernel)
                hasSEKernelPreloaded = isinstance(DSPunit.preloadedKernel, SEKernel)
                hasSEPreAdderLoaded = isinstance(DSPunit.preAdder, SEVectorChunk) and DSPunit.preAdder.valuesStored > 0
                hasSEPreAdderPreloaded = isinstance(DSPunit.preloadedPreAdder, SEVectorChunk) and DSPunit.preloadedPreAdder.valuesStored > 0

            if ((not hasSEKernelLoaded) and hasSEKernelPreloaded):
                DSPunit.loadKernel()
                hasSEKernelLoaded = True
                hasSEKernelPreloaded = False
            if ((not operation.isExpansion) and (not hasSEPreAdderLoaded) and hasSEPreAdderPreloaded):
                DSPunit.loadPreAdder()
                hasSEPreAdderLoaded = True
                hasSEPreAdderPreloaded = False

            if (hasSEKernelLoaded and hasSEKernelPreloaded):
                return False
            if ((not operation.isExpansion) and hasSEPreAdderLoaded and hasSEPreAdderPreloaded):
                return False

            maxWeightsToLoad = KERNEL_WEIGHT_CAP * 2
            if (hasSEKernelLoaded):
                maxWeightsToLoad = KERNEL_WEIGHT_CAP

            maxActivationsToLoad = maxWeightsToLoad
            if (not operation.isExpansion and hasSEPreAdderLoaded):
                maxActivationsToLoad = SE_VECTOR_CHUNK_PARTS

            weightsToLoad = min(remainingWeightsPerDSP, maxWeightsToLoad, maxActivationsToLoad)
            SEKernels = memory.getSEWeightChunksForDSP(
                operation.layer,
                operation.isExpansion,
                dspIndex,
                operation.weightsPerDSP,
                operation.weightsLoaded,
                weightsToLoad)
            if (SEKernels == False):
                operation.DSPindexStart = dspIndex
                return False

            SEVectorChunks = []
            if (not operation.isExpansion):
                SEVectorChunks = memory.getSEVectorChunksForDSP(
                    operation.layer,
                    dspIndex,
                    operation.weightsPerDSP,
                    operation.weightsLoaded,
                    weightsToLoad)

            if (hasSEKernelLoaded):
                DSPunit.preloadKernel(SEKernels[0])
                if (not operation.isExpansion):
                    DSPunit.preloadPreAdder(SEVectorChunks[0])
            else:
                DSPunit.preloadKernel(SEKernels[0])
                if (not operation.isExpansion):
                    DSPunit.preloadPreAdder(SEVectorChunks[0])
                DSPunit.loadKernel()
                if (not operation.isExpansion):
                    DSPunit.loadPreAdder()
                DSPunit.startDotProduct()
                self.recordSEAccumulatorStart(DSPunit, operation, dspIndex)
                if (len(SEKernels) > 1):
                    DSPunit.preloadKernel(SEKernels[1])
                    if (not operation.isExpansion):
                        DSPunit.preloadPreAdder(SEVectorChunks[1])

            operation.DSPindexStart = dspIndex + 1

        operation.weightsLoaded += weightsToLoad
        operation.DSPindexStart = 0
        if (operation.weightsLoaded < operation.weightsPerDSP):
            return False
        return self.extractSEVectorOutputs(operation, DSPunits)

    def startSEExciteOperation(self, operation: SEVectorProdOperation):
        squeezeDSPunits = self.getDSPsWithOpTag(operation.tag)
        for DSPunit in squeezeDSPunits:
            DSPunit.setOperationTag(None)

        operation.isExpansion = True
        operation.weightsLoaded = 0
        operation.DSPindexStart = 0
        operation.currentLoadingIndex = 0
        operation.currentUnloadingIndex = 0
        operation.currentOutputStoreIndex = 0
        operation.accumulatorRowsLinked = False
        operation.extractedOutputs.clear()

        freeDSPs = self.getFreeDSPsForSERows(
            operation.layer,
            operation.weightsPerDSP,
            True)
        expectedDSPCount = self.getSEDSPsNeededForMatrix(
            operation.layer,
            operation.weightsPerDSP,
            True)
        if (len(freeDSPs) != expectedDSPCount):
            return False

        operation.exciteTag = self.tagCounter
        self.assignOperationTag(freeDSPs, operation.exciteTag)
        self.tagCounter += 1
        operation.exciteOperationStarted = True
        return True

    def processSEExciteOperation(self, operation: SEVectorProdOperation):
        if (not operation.exciteOperationStarted):
            if (self.startSEExciteOperation(operation) == False):
                return False

        if (operation.exciteTag is None):
            raise Exception("ERROR! SE excite operation started without an excite tag")

        DSPunits = self.getDSPsWithOpTag(operation.exciteTag)
        expectedDSPCount = self.getSEDSPsNeededForMatrix(
            operation.layer,
            operation.weightsPerDSP,
            True)
        if (len(DSPunits) != expectedDSPCount):
            raise Exception("ERROR! SE excite operation must load all DSP units assigned to the operation")
        if (operation.weightsPerDSP < 0):
            raise Exception("ERROR! SE excite operation weightsPerDSP cannot be negative")
        if (operation.weightsLoaded < 0):
            raise Exception("ERROR! SE excite operation weightsLoaded cannot be negative")
        if (operation.weightsLoaded > operation.weightsPerDSP):
            raise Exception("ERROR! SE excite operation weightsLoaded cannot exceed weightsPerDSP")
        if (operation.DSPindexStart < 0 or operation.DSPindexStart > len(DSPunits)):
            raise Exception("ERROR! SE excite operation DSPindexStart is outside the DSP range")

        if (not operation.accumulatorRowsLinked):
            self.linkSEAccumulatorRows(operation, DSPunits)

        remainingWeightsPerDSP = operation.weightsPerDSP - operation.weightsLoaded
        if (remainingWeightsPerDSP == 0):
            return self.extractSEVectorOutputs(operation, DSPunits)

        for dspIndex in range(operation.DSPindexStart, len(DSPunits)):
            DSPunit = DSPunits[dspIndex]
            DSPunit.isActive()
            hasSEKernelLoaded = isinstance(DSPunit.kernel, SEKernel)
            hasSEKernelPreloaded = isinstance(DSPunit.preloadedKernel, SEKernel)
            hasSEPreAdderLoaded = isinstance(DSPunit.preAdder, SEVectorChunk) and DSPunit.preAdder.valuesStored > 0
            hasSEPreAdderPreloaded = isinstance(DSPunit.preloadedPreAdder, SEVectorChunk) and DSPunit.preloadedPreAdder.valuesStored > 0

            if ((not DSPunit.isActive()) and hasSEKernelLoaded and (not hasSEKernelPreloaded)):
                DSPunit.clearActiveStorage()
                hasSEKernelLoaded = False
                hasSEPreAdderLoaded = False

            if ((not DSPunit.isActive()) and DSPunit.hasPreloadedDotProduct()):
                DSPunit.startPreloadedDotProductIfAvailable()
                self.recordSEAccumulatorStart(DSPunit, operation, dspIndex)
                hasSEKernelLoaded = isinstance(DSPunit.kernel, SEKernel)
                hasSEKernelPreloaded = isinstance(DSPunit.preloadedKernel, SEKernel)
                hasSEPreAdderLoaded = isinstance(DSPunit.preAdder, SEVectorChunk) and DSPunit.preAdder.valuesStored > 0
                hasSEPreAdderPreloaded = isinstance(DSPunit.preloadedPreAdder, SEVectorChunk) and DSPunit.preloadedPreAdder.valuesStored > 0

            if ((not hasSEKernelLoaded) and hasSEKernelPreloaded):
                DSPunit.loadKernel()
                hasSEKernelLoaded = True
                hasSEKernelPreloaded = False
            if ((not hasSEPreAdderLoaded) and hasSEPreAdderPreloaded):
                DSPunit.loadPreAdder()
                hasSEPreAdderLoaded = True
                hasSEPreAdderPreloaded = False

            if (hasSEKernelLoaded and hasSEKernelPreloaded):
                return False
            if (hasSEPreAdderLoaded and hasSEPreAdderPreloaded):
                return False

            maxWeightsToLoad = KERNEL_WEIGHT_CAP * 2
            if (hasSEKernelLoaded):
                maxWeightsToLoad = KERNEL_WEIGHT_CAP

            maxActivationsToLoad = maxWeightsToLoad
            if (hasSEPreAdderLoaded):
                maxActivationsToLoad = SE_VECTOR_CHUNK_PARTS

            weightsToLoad = min(remainingWeightsPerDSP, maxWeightsToLoad, maxActivationsToLoad)
            SEKernels = memory.getSEWeightChunksForDSP(
                operation.layer,
                True,
                dspIndex,
                operation.weightsPerDSP,
                operation.weightsLoaded,
                weightsToLoad)
            if (SEKernels == False):
                operation.DSPindexStart = dspIndex
                return False

            SEHiddenLayerChunks = memory.getSEHiddenLayerChunksForDSP(
                operation.layer,
                dspIndex,
                operation.weightsPerDSP,
                operation.weightsLoaded,
                weightsToLoad)

            if (hasSEKernelLoaded):
                DSPunit.preloadKernel(SEKernels[0])
                DSPunit.preloadPreAdder(SEHiddenLayerChunks[0])
            else:
                DSPunit.preloadKernel(SEKernels[0])
                DSPunit.preloadPreAdder(SEHiddenLayerChunks[0])
                DSPunit.loadKernel()
                DSPunit.loadPreAdder()
                DSPunit.startDotProduct()
                self.recordSEAccumulatorStart(DSPunit, operation, dspIndex)
                if (len(SEKernels) > 1):
                    DSPunit.preloadKernel(SEKernels[1])
                    DSPunit.preloadPreAdder(SEHiddenLayerChunks[1])

            operation.DSPindexStart = dspIndex + 1

        operation.weightsLoaded += weightsToLoad
        operation.DSPindexStart = 0
        if (operation.weightsLoaded < operation.weightsPerDSP):
            return False
        return self.extractSEVectorOutputs(operation, DSPunits)
    
    def startDepthwiseConvFromCamera(self, targetX, targetY, camera: camera):
        lastNeededX = min(targetX + (kernelSize[0] // 2), camera.width - 1)
        lastNeededY = min(targetY + (kernelSize[0] // 2), camera.height - 1)
        lastNeededPixel = camera.xyToPixelCounter(lastNeededX, lastNeededY)

        if (camera.pixelNumber() < lastNeededPixel):
            raise Exception("ERROR! Camera pixel counter not advanced enough to do depthwise convolution")
        
        # Since layer 0 is where the camera output gets turned to 16 channels
        freeDSPGroups = self.getFreeAccumulatorDSPGroups(channelsOut[0])
        
        if (len(freeDSPGroups) != channelsOut[0]):
            return False

        freeDSPs = []
        for freeDSPGroup in freeDSPGroups:
            freeDSPs.extend(freeDSPGroup)
        
        self.assignOperationTag(freeDSPs, self.tagCounter)
        depthwiseConvOp = CameraDepthwiseConvOperation(tag=self.tagCounter, targetX=targetX, targetY=targetY)
        self.tagCounter += 1
        
        self.fifoPush(depthwiseConvOp)
        
        return True

    def startSEVectorCalc(self, layer):
        if (not pixelMap.isLayerFinished(layer)):
            # May want to remove this later to see how calculating SE vector with uncomplete averages
            # affects performance
            raise Exception("Trying to calculate SE Vector of unfinished layer")

        expectedDSPSpan = self.getSEDSPSpanForRows(layer, SEMACsPerDSP[layer], False)
        if (SEDSPsNeeded[layer] != expectedDSPSpan):
            raise Exception("ERROR! Squeeze DSPs Needed must include active squeeze DSPs and skipped DSP slots")

        freeDSPs = self.getFreeDSPsForSERows(layer, SEMACsPerDSP[layer], False)
        
        if (len(freeDSPs) != self.getSEDSPsNeededForMatrix(layer, SEMACsPerDSP[layer], False)):
            return False
            
        self.assignOperationTag(freeDSPs, self.tagCounter)
        SEVectorProdOp = SEVectorProdOperation(
            layer=layer,
            tag=self.tagCounter,
            weightsPerDSP=SEMACsPerDSP[layer])
        self.tagCounter += 1
        
        self.fifoPush(SEVectorProdOp)
        
        return True

    def processOperation(self):
        toCheck = self.fifoCheckLen()
        
        if (toCheck == 0):
            return
        
        fifoIndex = 0

        while (toCheck > 0):
            operation = self.fifoPeek(fifoIndex)
            
            # Depthwise convolutions go left to right, top to bottom
            INITIAL_CONVOLUTION_K = 3
            if isinstance(operation, CameraDepthwiseConvOperation):
                kernelRadius = INITIAL_CONVOLUTION_K // 2
                cameraActivationList = []
                for height in range(INITIAL_CONVOLUTION_K):
                    for width in range(INITIAL_CONVOLUTION_K):
                        activationX = operation.targetX + (width - kernelRadius)
                        activationY = operation.targetY + (height - kernelRadius)
                        if (activationX < 0 or activationX >= imxCamera.width):
                            continue
                        if (activationY < 0 or activationY >= imxCamera.height):
                            continue
                        cameraActivation = imxCamera.getActivation(activationX, activationY)
                        cameraActivationList.append(cameraActivation)
                activationChunk = memory.activationToActivationChunk(cameraActivationList)
                
                DSPunits = self.getDSPsWithOpTag(operation.tag)
                DSPGroups = [
                    DSPunits[index:index + DSPS_PER_ACCUMULATOR]
                    for index in range(0, len(DSPunits), DSPS_PER_ACCUMULATOR)
                ]
                
                if (operation.activationsLoaded == False):
                    channel = 0
                    for DSPGroup in DSPGroups:
                        DSPunit = DSPGroup[0]
                        DSPActivationChunk = replace(activationChunk, channel=channel)
                        DSPunit.preloadActivation(DSPActivationChunk)
                        channel += 1
                    operation.activationsLoaded = True
                
                if (len(DSPGroups) != channelsOut[0] or any(len(DSPGroup) != DSPS_PER_ACCUMULATOR for DSPGroup in DSPGroups)):
                    raise Exception("Camera Depthwise Conv Op attempted without one DSP accumulator group per output channel")
                
                if (operation.nextToRead == channelsOut[0]):
                    for DSPunit in DSPunits:
                        DSPunit.setOperationTag(None)
                    pixelMap.setPixelFinished(operation.targetX // stride[0], operation.targetY // stride[0], 0)
                    self.fifoPop()
                    fifoIndex -= 1
                    return
                
                if (operation.nextToRead < operation.nextToTrigger):
                    for channel in range(operation.nextToRead, operation.nextToTrigger):
                        accumulatorToRead = DSPGroups[channel][0].accumulator
                        
                        if (not accumulatorToRead.hasOutputReady()):
                            break
                        
                        # Load activation from DSP accumulator to SRAM
                        if (memory.loadActivationFromAccumulator(accumulatorToRead) == False):
                            break
                        
                        operation.nextToRead += 1
                
                if (operation.nextToTrigger < operation.nextToLoad):
                    for channel in range(operation.nextToTrigger, operation.nextToLoad):
                        DSPGroupToTrigger = DSPGroups[channel]
                        DSPtoTrigger = DSPGroupToTrigger[0]
                        
                        if (DSPtoTrigger.isActive()):
                            break
                        
                        DSPtoTrigger.loadKernel()
                        DSPtoTrigger.loadActivation()
                        DSPtoTrigger.startDotProduct()
                        operation.nextToTrigger += 1
                
                if (operation.nextToLoad != channelsOut[0]):
                    for channel in range(operation.nextToLoad, channelsOut[0]):
                        
                        desiredKernel = DepthwiseKernel(layer=0, channel=channel)

                        DSPunit = DSPGroups[channel][0]
                        result = memory.askForKernel(desiredKernel, DSPunit.getID())

                        if (result == False):
                            break
                        
                        operation.nextToLoad += 1

            if isinstance(operation, SEVectorProdOperation):
                if (not operation.squeezeOperationComplete):
                    squeezeOperationComplete = self.processSESqueezeOperation(operation)
                    if (squeezeOperationComplete):
                        operation.squeezeOperationComplete = True
                else:
                    exciteOperationComplete = self.processSEExciteOperation(operation)
                    if (exciteOperationComplete):
                        operation.exciteOperationComplete = True
                        pixelMap.setSEVectorCalculated(operation.layer)
                        if (operation.exciteTag is not None):
                            for DSPunit in self.getDSPsWithOpTag(operation.exciteTag):
                                DSPunit.setOperationTag(None)
                        self.fifoRemove(fifoIndex)
                        return
            
# @dataclass
# class PointwiseConvOperation:
#     type: Literal["pointwise"]
#     layer: int
#     tag: int
#     segment: int
#     outputChannel: int
#     step: int = 0
#     channelsCompleted: int = 0
            if (isinstance(operation, PointwiseConvOperation)):
                # desired behavior:
                # For the specified segment, each pixel in that segment should correspond to a DSP
                # accumulator.
                # The number of channels accumulated starts at 0
                # If (total_channels - channels_accumulated) is greater than 36, then load the next 36 indices
                # from the pointwise kernel, and load the next 36 channel activations from all of those pixels
                # If (total_channels - channels_accumulated) is less than 36, then load the remaining indices
                # from the pointwise kernel, and the remaining channel activations from all of those pixels
                
                # Now, 
                pass
                
                
                
                
            fifoIndex += 1
            toCheck -= 1
    
    def trackDSPActivations(self):
        if (not TRACKING_MODE_ON):
            return

        if (self.DSPActivationFirstClock is None):
            self.DSPActivationFirstClock = int(clkCycles)

        cycleIndex = self.DSPActivationCycleCount
        for dspIndex, dsp in enumerate(self.DSPlist):
            isActive = bool(dsp.isActive())
            if (isActive and self.DSPOpenRunStarts[dspIndex] is None):
                self.DSPOpenRunStarts[dspIndex] = cycleIndex
            elif ((not isActive) and self.DSPOpenRunStarts[dspIndex] is not None):
                self.DSPActiveRuns[dspIndex].append([self.DSPOpenRunStarts[dspIndex], cycleIndex])
                self.DSPOpenRunStarts[dspIndex] = None

        self.DSPActivationLastClock = int(clkCycles)
        self.DSPActivationCycleCount += 1

    def exportDSPActivationHistoryAsJSON(self):
        activeRuns = [[run[:] for run in dspRuns] for dspRuns in self.DSPActiveRuns]
        for dspIndex, runStart in enumerate(self.DSPOpenRunStarts):
            if (runStart is not None):
                activeRuns[dspIndex].append([runStart, self.DSPActivationCycleCount])

        return {
            "format": "active-runs-v1",
            "numDSPs": NUM_DSPS,
            "firstClock": self.DSPActivationFirstClock,
            "lastClock": self.DSPActivationLastClock,
            "numCycles": self.DSPActivationCycleCount,
            "activeRuns": activeRuns
        }

    def flushDSPActivationHistory(self):
        self.DSPActivationFirstClock = None
        self.DSPActivationLastClock = None
        self.DSPActivationCycleCount = 0
        self.DSPActiveRuns = [[] for _ in range(NUM_DSPS)]
        self.DSPOpenRunStarts = [None] * NUM_DSPS

    def update(self):
        self.processOperation()
        for accumulator in self.accumulators:
            accumulator.refreshLinkedOutputReadyClock()
        self.trackDSPActivations()

DSPMngr = DSPmanager()


NUM_LRAMS = 5
NUM_SRAMS = 166
MAX_SIMULTANEOUS_LRAM_WRITES = NUM_LRAMS
MAX_SIMULTANEOUS_SRAM_WRITES = NUM_SRAMS

WEIGHTS_PER_LRAM_ADDRESS = 4
SE_WEIGHTS_PER_SRAM_ADDRESS = 2
MAX_SQUEEZE_EXCITE_CHANNELS = 243
SRAM_ADDRESSES = 1024

# SRAM indices 0 through ACTIVATION_SRAM_START_INDEX - 1 are reserved for
# squeeze/excite weights and the maximum-size SE output vector. Activations use
# the remaining SRAM as a circular buffer. Going one value past the final SRAM
# location continues at ACTIVATION_SRAM_START_INDEX, not at physical index 0.
SRAM_TOTAL_VALUE_CAPACITY = NUM_SRAMS * SRAM_ADDRESSES * SE_WEIGHTS_PER_SRAM_ADDRESS
ACTIVATION_SRAM_START_INDEX = SEWEIGHTTOTALBYTES + MAX_SQUEEZE_EXCITE_CHANNELS
ACTIVATION_SRAM_CAPACITY = SRAM_TOTAL_VALUE_CAPACITY - ACTIVATION_SRAM_START_INDEX

def wrapActivationSRAMIndex(index: int) -> int:
    if (index < ACTIVATION_SRAM_START_INDEX):
        raise Exception("ERROR! Activation SRAM index overlaps reserved squeeze-excite storage")
    if (ACTIVATION_SRAM_CAPACITY <= 0):
        raise Exception("ERROR! No SRAM capacity remains for circular activation storage")

    return ACTIVATION_SRAM_START_INDEX + (
        (int(index) - ACTIVATION_SRAM_START_INDEX) % ACTIVATION_SRAM_CAPACITY)

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

        self.squeezeExciteAccumulator = np.zeros(MAX_SQUEEZE_EXCITE_CHANNELS)
        self.squeezeExciteAccumulatorCount = np.zeros(MAX_SQUEEZE_EXCITE_CHANNELS, dtype=int)
        self.SEHiddenLayerVector = SEHiddenLayerVector(length=0)
        
        self.operationFIFO = deque()
        self.memTagPushCounter = 0
        self.memTagPopCounter = 0

    def addActivationToSqueezeExciteAccumulator(self, activation: Activation):
        if (squeezeExcite[activation.layer] != 1):
            return
        if (activation.channel < 0 or activation.channel >= MAX_SQUEEZE_EXCITE_CHANNELS):
            raise Exception("ERROR! Activation channel is outside the squeeze-excite accumulator range")

        self.squeezeExciteAccumulator[activation.channel] += activation.value
        self.squeezeExciteAccumulatorCount[activation.channel] += 1

    def clearAccumulator(self):
        self.squeezeExciteAccumulator.fill(0)
        self.squeezeExciteAccumulatorCount.fill(0)

    def readAveragedChannel(self, layer, channel):
        if (layer < 0 or layer >= layers):
            raise Exception("ERROR! SE vector layer is outside the layer range")
        if (channel < 0 or channel >= channelsOut[layer]):
            raise Exception("ERROR! SE vector channel is outside the layer channel range")

        expectedAccumulations = spatialOutput[layer] * spatialOutput[layer]
        if (self.squeezeExciteAccumulatorCount[channel] != expectedAccumulations):
            raise Exception("ERROR! Requested SE vector channel is not fully accumulated")

        averageValue = self.squeezeExciteAccumulator[channel] / self.squeezeExciteAccumulatorCount[channel]
        return SEVectorIndex(type="input", layer=layer, channel=channel, value=np.uint8(averageValue))

    def SEVectorIndexToSEVectorChunk(self, SEVectorIndexList):
        if (len(SEVectorIndexList) <= 0 or len(SEVectorIndexList) > SE_VECTOR_CHUNK_PARTS):
            raise Exception("ERROR! SE vector chunk must be made from 1 to 9 SE vector indexes")
        if (any(SEVectorIndexObj is None for SEVectorIndexObj in SEVectorIndexList)):
            raise Exception("ERROR! Cannot make SE vector chunk from empty SE vector index")

        firstSEVectorIndex = SEVectorIndexList[0]
        if (not isinstance(firstSEVectorIndex, SEVectorIndex)):
            raise Exception("ERROR! SE vector chunk must be made from SE vector indexes")

        for part, SEVectorIndexObj in enumerate(SEVectorIndexList):
            if (not isinstance(SEVectorIndexObj, SEVectorIndex)):
                raise Exception("ERROR! SE vector chunk must be made from SE vector indexes")
            if (SEVectorIndexObj.type != firstSEVectorIndex.type):
                raise Exception("ERROR! SE vector chunk indexes must have the same type")
            if (SEVectorIndexObj.layer != firstSEVectorIndex.layer):
                raise Exception("ERROR! SE vector chunk indexes must be from the same layer")
            expectedChannel = (firstSEVectorIndex.channel + part) % channelsOut[firstSEVectorIndex.layer]
            if (SEVectorIndexObj.channel != expectedChannel):
                raise Exception("ERROR! SE vector chunk indexes are not sequential")

        return SEVectorChunk(
            type=firstSEVectorIndex.type,
            layer=firstSEVectorIndex.layer,
            channel=firstSEVectorIndex.channel,
            valuesStored=len(SEVectorIndexList))

    def setSEHiddenLayerVector(self, SEHiddenLayerVectorObj: SEHiddenLayerVector):
        if (not isinstance(SEHiddenLayerVectorObj, SEHiddenLayerVector)):
            raise Exception("ERROR! SE hidden layer vector must be an SEHiddenLayerVector")
        if (SEHiddenLayerVectorObj.length < 0):
            raise Exception("ERROR! SE hidden layer vector length cannot be negative")

        self.SEHiddenLayerVector = replace(SEHiddenLayerVectorObj)

    def getSEHiddenLayerVector(self):
        return replace(self.SEHiddenLayerVector)

    def readSEHiddenLayerElem(self, layer, elemNumber):
        if (layer < 0 or layer >= layers):
            raise Exception("ERROR! SE hidden layer elem layer is outside the layer range")
        if (elemNumber < 0 or elemNumber >= SEhiddenLayer[layer]):
            raise Exception("ERROR! SE hidden layer elem is outside the hidden layer range")
        if (self.SEHiddenLayerVector.length != SEhiddenLayer[layer]):
            raise Exception("ERROR! SE hidden layer vector is not ready for excite")

        return SEhiddenLayerElem(elemNumber=elemNumber)

    def SEhiddenLayerElemsToSEHiddenLayerVector(self, SEhiddenLayerElemList):
        if (len(SEhiddenLayerElemList) <= 0):
            raise Exception("ERROR! SE hidden layer vector must be made from at least one SE hidden layer elem")
        if (any(SEhiddenLayerElemObj is None for SEhiddenLayerElemObj in SEhiddenLayerElemList)):
            raise Exception("ERROR! Cannot make SE hidden layer vector from empty SE hidden layer elem")

        firstSEhiddenLayerElem = SEhiddenLayerElemList[0]
        if (not isinstance(firstSEhiddenLayerElem, SEhiddenLayerElem)):
            raise Exception("ERROR! SE hidden layer vector must be made from SE hidden layer elems")

        for part, SEhiddenLayerElemObj in enumerate(SEhiddenLayerElemList):
            if (not isinstance(SEhiddenLayerElemObj, SEhiddenLayerElem)):
                raise Exception("ERROR! SE hidden layer vector must be made from SE hidden layer elems")
            if (SEhiddenLayerElemObj.elemNumber != firstSEhiddenLayerElem.elemNumber + part):
                raise Exception("ERROR! SE hidden layer elems are not sequential")

        SEHiddenLayerVectorObj = SEHiddenLayerVector(length=len(SEhiddenLayerElemList))
        self.setSEHiddenLayerVector(SEHiddenLayerVectorObj)
        return SEHiddenLayerVectorObj

    def getSEWeightChunksForDSP(self, layer, isExpansion, dspIndex, weightsPerDSP, weightsLoaded, weightsToLoad):
        if (weightsPerDSP < 0):
            raise Exception("ERROR! weightsPerDSP cannot be negative")
        if (weightsLoaded < 0):
            raise Exception("ERROR! weightsLoaded cannot be negative")
        if (weightsToLoad < 0):
            raise Exception("ERROR! weightsToLoad cannot be negative")
        if (dspIndex < 0 or dspIndex >= NUM_DSPS):
            raise Exception("ERROR! DSP index is outside the DSP range")

        inputVectorLength = SEhiddenLayer[layer] if isExpansion else channelsOut[layer]
        if (inputVectorLength <= 0):
            raise Exception("ERROR! SE input vector length must be positive")

        DSPsPerRow = math.ceil(inputVectorLength / weightsPerDSP)
        row = dspIndex // DSPsPerRow
        dspIndexInRow = dspIndex % DSPsPerRow

        layerStart = self.startOfSELayer[layer]
        matrixOffset = 0
        if (isExpansion):
            matrixOffset = channelsOut[layer] * SEhiddenLayer[layer]

        trueStartOffset = matrixOffset + (row * inputVectorLength) + (dspIndexInRow * weightsPerDSP) + weightsLoaded
        startLocation = self.offsetSESRAMLocation(
            layerStart[0],
            layerStart[1],
            layerStart[2],
            trueStartOffset)

        weights = []
        success = self.readSEWeightsSequentially(
            startLocation[0],
            startLocation[1],
            startLocation[2],
            weightsToLoad,
            weights)

        if (success == False):
            return False
        if (any(weight is None for weight in weights)):
            raise Exception("ERROR! Cannot load empty SE weight into DSP")

        weightChunks = [
            weights[index:index + KERNEL_WEIGHT_CAP]
            for index in range(0, len(weights), KERNEL_WEIGHT_CAP)
        ]
        if (len(weightChunks) == 0):
            weightChunks.append([])

        return [SEKernel(weights=weightChunk) for weightChunk in weightChunks]

    def getSEVectorChunksForDSP(self, layer, dspIndex, valuesPerDSP, valuesLoaded, valuesToLoad):
        if (valuesPerDSP < 0):
            raise Exception("ERROR! valuesPerDSP cannot be negative")
        if (valuesLoaded < 0):
            raise Exception("ERROR! valuesLoaded cannot be negative")
        if (valuesToLoad < 0):
            raise Exception("ERROR! valuesToLoad cannot be negative")
        if (dspIndex < 0 or dspIndex >= NUM_DSPS):
            raise Exception("ERROR! DSP index is outside the DSP range")

        inputVectorLength = channelsOut[layer]
        if (inputVectorLength <= 0):
            raise Exception("ERROR! SE input vector length must be positive")

        DSPsPerRow = math.ceil(inputVectorLength / valuesPerDSP)
        dspIndexInRow = dspIndex % DSPsPerRow

        SEVectorIndexes = []
        activationStartChannel = (dspIndexInRow * valuesPerDSP + valuesLoaded) % channelsOut[layer]
        for activationOffset in range(valuesToLoad):
            activationChannel = (activationStartChannel + activationOffset) % channelsOut[layer]
            SEVectorIndexes.append(self.readAveragedChannel(layer, activationChannel))

        SEVectorIndexChunks = [
            SEVectorIndexes[index:index + SE_VECTOR_CHUNK_PARTS]
            for index in range(0, len(SEVectorIndexes), SE_VECTOR_CHUNK_PARTS)
        ]

        return [
            self.SEVectorIndexToSEVectorChunk(SEVectorIndexChunk)
            for SEVectorIndexChunk in SEVectorIndexChunks
        ]

    def getSEHiddenLayerChunksForDSP(self, layer, dspIndex, valuesPerDSP, valuesLoaded, valuesToLoad):
        if (valuesPerDSP < 0):
            raise Exception("ERROR! valuesPerDSP cannot be negative")
        if (valuesLoaded < 0):
            raise Exception("ERROR! valuesLoaded cannot be negative")
        if (valuesToLoad < 0):
            raise Exception("ERROR! valuesToLoad cannot be negative")
        if (dspIndex < 0 or dspIndex >= NUM_DSPS):
            raise Exception("ERROR! DSP index is outside the DSP range")

        inputVectorLength = SEhiddenLayer[layer]
        if (inputVectorLength <= 0):
            raise Exception("ERROR! SE hidden layer vector length must be positive")

        DSPsPerRow = math.ceil(inputVectorLength / valuesPerDSP)
        dspIndexInRow = dspIndex % DSPsPerRow

        SEhiddenLayerElems = []
        elemStart = (dspIndexInRow * valuesPerDSP + valuesLoaded) % inputVectorLength
        for elemOffset in range(valuesToLoad):
            elemNumber = (elemStart + elemOffset) % inputVectorLength
            SEhiddenLayerElems.append(self.readSEHiddenLayerElem(layer, elemNumber))

        SEhiddenLayerElemChunks = [
            SEhiddenLayerElems[index:index + SE_VECTOR_CHUNK_PARTS]
            for index in range(0, len(SEhiddenLayerElems), SE_VECTOR_CHUNK_PARTS)
        ]

        return [
            SEVectorChunk(
                type="input",
                layer=layer,
                channel=SEhiddenLayerElemChunk[0].elemNumber,
                valuesStored=len(SEhiddenLayerElemChunk))
            for SEhiddenLayerElemChunk in SEhiddenLayerElemChunks
        ]

    def loadSEVectorOutput(self, SEVectorIndexObj: SEVectorIndex):
        if (not isinstance(SEVectorIndexObj, SEVectorIndex)):
            raise Exception("ERROR! SE vector output must be an SEVectorIndex")
        if (SEVectorIndexObj.type != "output"):
            raise Exception("ERROR! Only SE output vector indexes can be stored as excite output")

        return self.loadSEVectorPart(SEVectorIndexObj)

    def getSEVectorSRAMLocation(self, SEVectorIndex: SEVectorIndex) -> tuple[int, int, int]:
        if (SEVectorIndex.channel < 0 or SEVectorIndex.channel >= MAX_SQUEEZE_EXCITE_CHANNELS):
            raise Exception("ERROR! SE vector channel is outside the SE vector SRAM range")

        trueIndex = SEWEIGHTTOTALBYTES + SEVectorIndex.channel
        packedIndex = trueIndex % SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMJumps = trueIndex // SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMNum = SRAMJumps % NUM_SRAMS
        SRAMAddr = SRAMJumps // NUM_SRAMS

        return (SRAMNum, SRAMAddr, packedIndex)

    def loadSEVectorPart(self, SEVectorIndex: SEVectorIndex):
        SEVectorLocation = self.getSEVectorSRAMLocation(SEVectorIndex)

        SRAMNum = SEVectorLocation[0]
        SRAMAddr = SEVectorLocation[1]
        packedIndex = SEVectorLocation[2]
        SRAMobj = self.allSRAM[SRAMNum]

        if (SRAMobj.isActiveRead() or SRAMobj.isActiveWrite()):
            return False

        packedSEVectorParts = list(SRAMobj.readSRAM(SRAMAddr))
        packedSEVectorParts[packedIndex] = SEVectorIndex
        SRAMobj.writeSRAM(SRAMAddr, packedSEVectorParts)

        return True
    
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

    def readSEWeightsSequentially(self, SRAMNum, SRAMAddr, packedIndex, numReads, weightsRead):
        if (packedIndex < 0 or packedIndex >= SE_WEIGHTS_PER_SRAM_ADDRESS):
            raise Exception("ERROR! packedIndex is outside the SRAM packed address")
        if (numReads < 0):
            raise Exception("ERROR! numReads cannot be negative")

        currentSRAMNum = SRAMNum
        currentAddr = SRAMAddr
        currentIndex = packedIndex

        while (len(weightsRead) < numReads):
            SRAMobj = self.allSRAM[currentSRAMNum]

            if (SRAMobj.isActiveRead()):
                return False

            fullRead = SRAMobj.readSRAM(currentAddr)

            while (currentIndex < SE_WEIGHTS_PER_SRAM_ADDRESS and len(weightsRead) < numReads):
                weightsRead.append(fullRead[currentIndex])
                currentIndex += 1

            currentIndex = 0
            currentSRAMNum += 1
            if (currentSRAMNum >= NUM_SRAMS):
                currentSRAMNum = 0
                currentAddr += 1

        return True

    def offsetSESRAMLocation(self, SRAMNum, SRAMAddr, packedIndex, offset):
        if (packedIndex < 0 or packedIndex >= SE_WEIGHTS_PER_SRAM_ADDRESS):
            raise Exception("ERROR! packedIndex is outside the SRAM packed address")
        if (offset < 0):
            raise Exception("ERROR! SRAM offset cannot be negative")

        totalPackedIndex = packedIndex + offset
        SRAMJumps = totalPackedIndex // SE_WEIGHTS_PER_SRAM_ADDRESS
        newPackedIndex = totalPackedIndex % SE_WEIGHTS_PER_SRAM_ADDRESS

        logicalSRAM = SRAMNum + SRAMJumps
        newSRAMNum = logicalSRAM % NUM_SRAMS
        newSRAMAddr = SRAMAddr + (logicalSRAM // NUM_SRAMS)

        return (newSRAMNum, newSRAMAddr, newPackedIndex)

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
            layerSpatialSize = spatialInput[firstActivation.layer]

            if (layerKernelSize != 3 and layerKernelSize != 5):
                raise Exception("ERROR! Unsupported depthwise activation kernel size")

            for activation in activationList:
                if (activation.layer != firstActivation.layer or activation.channel != firstActivation.channel):
                    raise Exception("ERROR! Depthwise activations are not in the same chunk")
                if (activation.x < 0 or activation.x >= layerSpatialSize):
                    raise Exception(f"ERROR! Depthwise activation x coordinate is outside the layer bounds.\n X val = {activation.x}")
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
    
    def loadActivationsFromAccumulators(self, accumulatorWithActivation, accumulatorWithActivation2):
        if ((not accumulatorWithActivation.hasOutputReady()) or (not accumulatorWithActivation2.hasOutputReady())):
            return False

        # Now, I need to come up with a scheme on how activations should be loaded and fetched from memory
        activation1 = accumulatorWithActivation.getOutput()
        activation2 = accumulatorWithActivation2.getOutput()
        
        if (activation1.layer != activation2.layer):
            raise Exception("Activations do not come from the same layer!")
        
        if (layerList[activation1.layer + 1] == "depthwise"):
            return False
        elif (layerList[activation1.layer + 1] == "pointwise"):
            writeSuccessful = self.loadActivationsToSRAM(activation1, activation2)
        else:
            raise Exception("Loading activation from DSP to non pointwise or depthwise layers not supported... for now")

        if (not writeSuccessful):
            return False

        accumulatorWithActivation.clearOutput()
        accumulatorWithActivation2.clearOutput()
        return True

    def loadActivationsFromDSPs(self, DSPWithActivation, DSPWithActivation2):
        return self.loadActivationsFromAccumulators(
            DSPWithActivation.accumulator,
            DSPWithActivation2.accumulator)

    def loadActivationFromAccumulator(self, accumulatorWithActivation):
        if (not accumulatorWithActivation.hasOutputReady()):
            return False

        activation = accumulatorWithActivation.getOutput()

        if (layerList[activation.layer + 1] == "depthwise"):
            return False
        elif (layerList[activation.layer + 1] == "pointwise"):
            writeSuccessful = self.loadActivationToSRAM(activation)
        else:
            raise Exception("Loading activation from DSP to non pointwise or depthwise layers not supported... for now")

        if (not writeSuccessful):
            return False

        accumulatorWithActivation.clearOutput()
        return True

    def loadActivationFromDSP(self, DSPWithActivation):
        return self.loadActivationFromAccumulator(DSPWithActivation.accumulator)

    def getActivationSRAMLocation(self, activation):
        layer = activation.layer
        numChannels = channelsOut[layer]
        maxX = spatialOutput[layer]

        if (activation.channel < 0 or activation.channel >= numChannels):
            raise Exception("ERROR! Activation channel is outside the layer channel range")
        if (activation.x < 0 or activation.x >= maxX):
            raise Exception(f"ERROR! Activation x coordinate is outside the layer spatial range (X = {activation.x}, maxX = {maxX})")
        if (activation.y < 0 or activation.y >= maxX):
            raise Exception("ERROR! Activation y coordinate is outside the layer spatial range")

        index = activation.channel + (activation.x * numChannels) + (maxX * numChannels * activation.y)
        unwrappedIndex = index + SRAMStartIndex[layer]
        trueIndex = wrapActivationSRAMIndex(unwrappedIndex)
        packedIndex = trueIndex % SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMJumps = trueIndex // SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMNum = SRAMJumps % NUM_SRAMS
        SRAMAddr = SRAMJumps // NUM_SRAMS

        return (SRAMNum, SRAMAddr, packedIndex)
    
    def loadActivationsToSRAM(self, activation1, activation2):
        if (activation1.layer != activation2.layer):
            raise Exception("ERROR! Activations stored together must come from the same layer")

        activation1Location = self.getActivationSRAMLocation(activation1)
        activation2Location = self.getActivationSRAMLocation(activation2)

        if (activation1Location == activation2Location):
            raise Exception("ERROR! Activations cannot use the same packed SRAM position")

        activationLocations = [activation1Location, activation2Location]
        SRAMWords = {(location[0], location[1]) for location in activationLocations}

        # The circular boundary can split two consecutive activations across two
        # physical SRAM words. Both writes can still occur in one simulated cycle
        # when those words live in different SRAM blocks.
        if (len(SRAMWords) > 1):
            if (activation1Location[0] == activation2Location[0]):
                raise Exception("ERROR! Cannot write two addresses in one SRAM during the same clock cycle")

            for SRAMNum, _, _ in activationLocations:
                SRAMobj = self.allSRAM[SRAMNum]
                if (SRAMobj.isActiveRead() or SRAMobj.isActiveWrite()):
                    return False

            for activation, (SRAMNum, SRAMAddr, packedIndex) in zip(
                    (activation1, activation2), activationLocations):
                SRAMobj = self.allSRAM[SRAMNum]
                packedActivations = list(SRAMobj.readSRAM(SRAMAddr))
                packedActivations[packedIndex] = activation
                SRAMobj.writeSRAM(SRAMAddr, packedActivations)
        else:
            SRAMNum = activation1Location[0]
            SRAMAddr = activation1Location[1]
            SRAMobj = self.allSRAM[SRAMNum]

            if (SRAMobj.isActiveWrite()):
                return False

            packedActivations = [None] * SE_WEIGHTS_PER_SRAM_ADDRESS
            packedActivations[activation1Location[2]] = activation1
            packedActivations[activation2Location[2]] = activation2
            SRAMobj.writeSRAM(SRAMAddr, packedActivations)

        self.addActivationToSqueezeExciteAccumulator(activation1)
        self.addActivationToSqueezeExciteAccumulator(activation2)

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
        self.addActivationToSqueezeExciteAccumulator(activation)

        return True

    def pushMemoryOp(self, memoryOp: MemoryOperation):
        self.operationFIFO.appendleft(memoryOp)
    
    def popMemoryOp(self) -> MemoryOperation:
        return self.operationFIFO.pop()
    
    def peekMemoryOp(self, distanceFromTop) -> MemoryOperation:
        index = -(distanceFromTop + 1)
        return self.operationFIFO[index]
    
    def processMemoryOp(self):
        
        memoryOp = self.peekMemoryOp(0)
        
        if (isinstance(memoryOp, LoadSEWeightsMemOp)):
            raise Exception("ERROR! LoadSEWeightsMemOp has moved into SEVectorProdOperation in the DSP operation queue.")
            
def exportToJSON(memoryManagerObj, message="", rootDirectory="snapshots"):
    rootPath = Path(rootDirectory)
    clockCycleDirectory = rootPath / str(clkCycles)

    SRAMDirectory = clockCycleDirectory / "SRAM"
    SRAMDirectory.mkdir(parents=True, exist_ok=True)
    for SRAMIndex, SRAMobj in enumerate(memoryManagerObj.allSRAM):
        filePath = SRAMDirectory / f"SRAM{SRAMIndex}.json"
        with filePath.open("w", encoding="utf-8") as jsonFile:
            json.dump(SRAMobj.exportAsJSON(), jsonFile, indent=2)

    LRAMDirectory = clockCycleDirectory / "LRAM"
    LRAMDirectory.mkdir(parents=True, exist_ok=True)
    for LRAMIndex, LRAMobj in enumerate(memoryManagerObj.allLRAM):
        filePath = LRAMDirectory / f"LRAM{LRAMIndex}.json"
        with filePath.open("w", encoding="utf-8") as jsonFile:
            json.dump(LRAMobj.exportAsJSON(), jsonFile, indent=2)

    with (clockCycleDirectory / "message.json").open("w", encoding="utf-8") as jsonFile:
        json.dump({"message": message}, jsonFile, indent=2)

    if (TRACKING_MODE_ON):
        with (clockCycleDirectory / "DSPActivations.json").open("w", encoding="utf-8") as jsonFile:
            json.dump(DSPMngr.exportDSPActivationHistoryAsJSON(), jsonFile, separators=(",", ":"))
        DSPMngr.flushDSPActivationHistory()

    rootPath.mkdir(parents=True, exist_ok=True)
    startOfLayerData = []
    for layer in range(len(memoryManagerObj.startOfLayer)):
        startOfLayerData.append({
            "layer": layer,
            "LRAMNum": int(memoryManagerObj.startOfLayer[layer][0]),
            "LRAMAddr": int(memoryManagerObj.startOfLayer[layer][1]),
            "packedPosition": int(memoryManagerObj.startOfLayer[layer][2])
        })

    with (rootPath / "LRAMStartOfLayer.json").open("w", encoding="utf-8") as jsonFile:
        json.dump(startOfLayerData, jsonFile, indent=2)

    SRAMStartData = []
    for layer in range(len(SRAMStartIndex)):
        trueIndex = int(SRAMStartIndex[layer])
        packedIndex = trueIndex % SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMJumps = trueIndex // SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMNum = SRAMJumps % NUM_SRAMS
        SRAMAddr = SRAMJumps // NUM_SRAMS
        SRAMStartData.append({
            "layer": layer,
            "SRAMStartIndex": trueIndex,
            "SRAMNum": int(SRAMNum),
            "SRAMAddr": int(SRAMAddr),
            "packedIndex": int(packedIndex)
        })

    with (rootPath / "SRAMStartOfLayer.json").open("w", encoding="utf-8") as jsonFile:
        json.dump(SRAMStartData, jsonFile, indent=2)

    return clockCycleDirectory

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
    
    segment_pixels = spatialOutput[layer] ** 2 // 4
    input_chunks = channelsIn[layer] // POINTWISE_KERNEL_PARTS
    
    if (input_chunks % 4 != 0):
        adj_input_chunks = input_chunks + (4 - input_chunks % 4)
    else:
        adj_input_chunks = input_chunks
    
    one_shots_spatial_segment = (
        segment_pixels * adj_input_chunks < NUM_DSPS
    )

    if (one_shots_spatial_segment and layerList[layer] == "pointwise"):
        print(f"segment_pixels is {segment_pixels}")
        print(f"input_chunks is {input_chunks}")
        print(f"adj_input_chunks is {adj_input_chunks}")
        raise Exception(f"Layer is {layer}")
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
                
                if (channel >= channelsOut[layer]):
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

# Code below: stores the squeeze excite neural net
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

inferenceCompleted = False

imageJustIn = True

SIM_LIMIT = 1000000

while (inferenceCompleted == False and not clkCycles > SIM_LIMIT):
    
    # Special first layer logic
    if (pixelsComputed < spatialOutput[0] ** 2):
        imxCamera.updateCamera()
        
        pixelNumber = imxCamera.pixelNumber()
        if (pixelNumber < 0):
            pass
        else:
            # This is the next input pixel the stride lands on.
            targetOutputX = pixelsComputed % spatialOutput[0]
            targetOutputY = pixelsComputed // spatialOutput[0]

            targetX = targetOutputX * stride[0]
            targetY = targetOutputY * stride[0]

            kernelRadius = kernelSize[0] // 2
            lastNeededX = min(targetX + kernelRadius, imxCamera.width - 1)
            lastNeededY = min(targetY + kernelRadius, imxCamera.height - 1)
            lastNeededPixel = imxCamera.xyToPixelCounter(lastNeededX, lastNeededY)
            
            print(f"cycles: {clkCycles}, targetX: {targetX}, targetY: {targetY}, pixelCounter: {pixelNumber}, pixelXY: {imxCamera.pixelCounterToXY(pixelNumber)}\n")

            if (pixelNumber < lastNeededPixel):
                pass
            else:
                if (DSPMngr.startDepthwiseConvFromCamera(targetX, targetY, imxCamera)):
                    pixelsComputed += 1
    else:
        break
        if (imageJustIn):
            exportToJSON(memory, "image just recieved")
            imageJustIn = False
    
    # Universal logic
    # Write code that gets the last completed layer from the pixel historian,
    # If the last completed layer is an SE layer, then check whether the SE vector for that layer was completed.
    #   If yes, then, load the SE vector into the DSP pre-adders appropriately (additive attention, need to add
    #   support for pre-adding in the DSP units).
    #   If no, then, do the SE layer
    # If the last completed layer is not an SE layer, then just finish the next layer as soon as possible.
    # And queue up operations for the layer after that.
    # I seem to be forgetting that the job of the global controller is just to fill up a FIFO queue. It should have
    # no knowledge of compute resources, that's the DSPManager's job.
    
    lastCompletedLayer = pixelMap.getLastCompletedLayer()
    nextLayer = lastCompletedLayer + 1
    
    if (squeezeExcite[lastCompletedLayer] and not pixelMap.isSEVectorCalculated(lastCompletedLayer)):
        if (not DSPMngr.isSEVectorCalculating()):
            DSPMngr.startSEVectorCalc(lastCompletedLayer)
        elif (pixelMap.isSEVectorCalculated(lastCompletedLayer)):
            pass
            # I think the only way to really implement attention efficiently is with bit-shifts
            # There is no activation after attention is applied
    else:
        if (layerType[nextLayer] == "pointwise"):
            if (not pixelMap.isQuarterFinished(nextLayer, 1)):
                pass
            elif (not pixelMap.isQuarterFinished(nextLayer, 2)):
                pass
            elif (not pixelMap.isQuarterFinished(nextLayer, 3)):
                pass
            elif (not pixelMap.isQuarterFinished(nextLayer, 4)):
                pass
            # If segment 1 is not completed, do segment 1
            # If segment 2 is not completed, do segment 2
            # If segment 3 is not completed, do segment 3
            # If segment 4 is not completed, do segment 4
    
    # Fully Connected Neural Net Logic
    
    DSPMngr.update()
    clkCycles += 1

exportToJSON(memory, "done with inference")
