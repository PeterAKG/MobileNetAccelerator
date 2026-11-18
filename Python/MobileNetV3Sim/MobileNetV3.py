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
SEMACsPerDSP = np.zeros(layers, dtype=int)
SEExciteMACsPerDSP = np.zeros(layers, dtype=int)
SEExciteOutputChannelDivisor = np.ones(layers, dtype=int)
pointwiseQuarterDivision = np.ones(layers, dtype=int)
pointwiseSRAMScheme = np.zeros(layers, dtype="U20")
SEWEIGHTTOTALBYTES = int(df.loc[35, "SE Total NN Size (Bytes)"])
FULLY_CONNECTED_INPUT = int(df.loc[36, "In"])
FULLY_CONNECTED_HIDDEN = int(df.loc[36, "Out"])
FULLY_CONNECTED_OUTPUT = int(df.loc[37, "Out"])
FULLY_CONNECTED_LAYER_START = layers - NUM_MISC_LAYERS + 1
GAP_INPUT_LAYER = FULLY_CONNECTED_LAYER_START - 2
fullyConnectedInputDivisor = np.zeros(2, dtype=int)
fullyConnectedInputDivisor[0] = int(df.loc[36, "Fully Connected Divided Input"])
fullyConnectedInputDivisor[1] = int(df.loc[37, "Fully Connected Divided Input"])
fullyConnectedMACsPerDSP = np.zeros(2, dtype=int)
fullyConnectedMACsPerDSP[0] = int(df.loc[36, "Fully Connected MACs / DSP"])
fullyConnectedMACsPerDSP[1] = int(df.loc[37, "Fully Connected MACs / DSP"])

clkCycles = 0

POINTWISE_QUARTER_DIVISION_COLUMN = "Pointwise Divisions Per Quarter"
DEPTHWISE_KERNEL_SIZE = 3

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
        if (layerType == "depthwise" and kernelSize[idx] != DEPTHWISE_KERNEL_SIZE):
            raise Exception(
                f"ERROR! Depthwise layer {idx} has unsupported "
                f"{kernelSize[idx]}x{kernelSize[idx]} kernel; only 3x3 is supported")
        stride[idx] = int(row["Stride"])
        channelsIn[idx] = int(row["In"])
        channelsOut[idx] = int(row["Out"])
        spatialOutput[idx] = int(row["Spatial Output"])
        spatialInput[idx] = int(row["Spatial Input"])
        if(row["SE"] == "Yes"):
            squeezeExcite[idx] = 1
            SEhiddenLayer[idx] = int(row["SE Hidden Layer"])
            SEMACsPerDSP[idx] = int(row["SE Squeeze MACs / DSP"])
            SEExciteMACsPerDSP[idx] = int(row["Output SE MACs/DSP"])
            SEExciteOutputChannelDivisor[idx] = int(row["Excite Operation Output Channel Dividor"])
            if (SEMACsPerDSP[idx] <= 0 or SEExciteMACsPerDSP[idx] <= 0):
                raise Exception(f"ERROR! SE layer {idx} must use positive MACs per DSP")
            if (
                    SEExciteOutputChannelDivisor[idx] <= 0
                    or SEExciteOutputChannelDivisor[idx] > channelsOut[idx]):
                raise Exception(f"ERROR! SE layer {idx} has an invalid excite output-channel divisor")

        if(row["Layer Type"] == "pointwise"):
            # Spreadsheet formulas can produce a fraction once an entire quarter
            # already fits in the DSP array. Such a quarter is not subdivided.
            pointwiseQuarterDivision[idx] = max(
                1,
                int(row["Pointwise Divisions Per Quarter"])
            )
            pointwiseSRAMScheme[idx] = row["SRAM Loading Scheme"]

        SRAMStartIndex[idx] = int(row["SRAM Activation Start Index"])

for fullyConnectedStage in range(2):
    fullyConnectedOutputLength = (
        FULLY_CONNECTED_HIDDEN
        if fullyConnectedStage == 0
        else FULLY_CONNECTED_OUTPUT)
    if (fullyConnectedInputDivisor[fullyConnectedStage] <= 0
            or fullyConnectedInputDivisor[fullyConnectedStage] > fullyConnectedOutputLength):
        raise Exception(f"ERROR! Fully connected stage {fullyConnectedStage} has an invalid output divisor")
    if (fullyConnectedMACsPerDSP[fullyConnectedStage] <= 0):
        raise Exception(f"ERROR! Fully connected stage {fullyConnectedStage} must use positive MACs per DSP")

DSP_CHUNK_SIZE = 8
POINTWISE_VALUES_PER_DSP = 2
DEPTHWISE_KERNEL_PARTS = DEPTHWISE_KERNEL_SIZE ** 2

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
    kernelParts: tuple[int, ...] = field(default_factory=lambda: tuple(range(DEPTHWISE_KERNEL_PARTS)))
    valuesStored: int = DEPTHWISE_KERNEL_PARTS

@dataclass
class PointwiseWeight:
    type: Literal["pointwise"] = field(default="pointwise", init=False)
    layer: int = -1
    y: int = -1
    channel: int = -1

@dataclass
class PointwiseKernel:
    type: Literal["pointwise"] = field(default="pointwise", init=False)
    layer: int
    channel: int # Output channel.
    inputChannels: tuple[int, ...]

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
        if (len(self.weights) > DSP_CHUNK_SIZE):
            raise Exception(f"ERROR! SE kernel cannot contain more than {DSP_CHUNK_SIZE} weights")

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
    kernelParts: tuple[int, ...] = field(default_factory=tuple)
    valuesStored: int = 0
    inputChannels: tuple[int, ...] = field(default_factory=tuple)

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
    channel: int = -1   # First channel in this vector chunk.
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
    type: Literal["depthwise"] = field(default="depthwise", init=False)
    layer: int
    tag: int
    supervisorTag: int
    stride: int
    quarter: int
    channel: int
    numDSPGroups: int
    step: int = 0
    batchStartPixel: int = 0
    nextDSPToLoad: int = 0
    nextOutputToStore: int = 0
    bufferedKernel: Optional[DepthwiseKernel] = None # This is dumb bro. DepthwiseConvOperation should not have "BufferedKernel"
    kernelDistributedForBatch: bool = False

@dataclass
class DepthwiseSupervisorOperation:
    type: Literal["depthwiseSupervisor"] = field(default="depthwiseSupervisor", init=False)
    layer: int
    tag: int
    quarter: int
    nextChannel: int = 0
    childrenInFlight: int = 0
    childrenCompleted: int = 0

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
class PointwiseGroupState:
    channelsCompleted: int = 0
    channelsInFlight: int = 0
    outputStored: bool = False

@dataclass
class PointwiseConvOperation:
    type: Literal["pointwise"] = field(default="pointwise", init=False)
    layer: int
    tag: int
    supervisorTag: int
    quarter: int
    quarterSegment: int
    outputChannel: int
    groupStates: List[PointwiseGroupState]
    pixels: Optional[tuple[tuple[int, int], ...]] = None

@dataclass
class PointwiseSupervisorOperation:
    type: Literal["pointwiseSupervisor"] = field(default="pointwiseSupervisor", init=False)
    layer: int
    tag: int
    quarter: int
    completedQuarters: tuple[int, ...] = ()
    pendingQuarters: tuple[int, ...] = ()
    pixelSegments: tuple[tuple[tuple[int, int], ...], ...] = ()
    quarterSegment: int = 0
    nextOutputChannel: int = 0
    childrenInFlight: int = 0
    childrenCompleted: int = 0

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
    exciteBatchIndex: int = 0

@dataclass
class FullyConnectedOperation:
    type: Literal["fullyConnected"] = field(default="fullyConnected", init=False)
    tag: int
    stage: int = 0
    batchIndex: int = 0
    batchStarted: bool = False
    batchTag: Optional[int] = None
    weightsLoaded: int = 0
    DSPindexStart: int = 0
    currentUnloadingIndex: int = 0
    currentOutputStoreIndex: int = 0
    accumulatorRowsLinked: bool = False
    extractedOutputs: List[SEVectorIndex] = field(default_factory=list)

Operation = Union[
    DepthwiseConvOperation,
    DepthwiseSupervisorOperation,
    CameraDepthwiseConvOperation,
    PointwiseConvOperation,
    PointwiseSupervisorOperation,
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
SEGMENTS_PER_LAYER = 4
# This tracks the OUTPUT of each respective layer. Not the INPUT.
class pixelHistorian:
    
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.pixelHistoryMap = np.zeros((width, height, depth), dtype=bool)
        self.SEVectorCalculated = np.zeros(depth, dtype=bool)
        self.completedSEVectorLayers = set()
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

        self.advanceCompletedLayers()

    def advanceCompletedLayers(self):

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

    def getQuarterBounds(self, layer, quarter):
        layerSpatialSize = spatialOutput[layer]
        halfSpatialSize = layerSpatialSize // 2

        if (quarter == 1):
            xStart = 0
            xEnd = halfSpatialSize
            yStart = 0
            yEnd = halfSpatialSize
        elif (quarter == 2):
            xStart = halfSpatialSize
            xEnd = layerSpatialSize
            yStart = 0
            yEnd = halfSpatialSize
        elif (quarter == 3):
            xStart = 0
            xEnd = halfSpatialSize
            yStart = halfSpatialSize
            yEnd = layerSpatialSize
        elif (quarter == 4):
            xStart = halfSpatialSize
            xEnd = layerSpatialSize
            yStart = halfSpatialSize
            yEnd = layerSpatialSize
        else:
            raise Exception("ERROR! Quarter must be 1, 2, 3, or 4")

        return (xStart, xEnd, yStart, yEnd)

    def setQuarterFinished(self, layer, quarter):
        layerSlot = self.getLayerSlot(layer)
        xStart, xEnd, yStart, yEnd = self.getQuarterCompletionBounds(layer, quarter)
        self.pixelHistoryMap[xStart:xEnd, yStart:yEnd, layerSlot] = True
        self.advanceCompletedLayers()

    def isQuarterFinished(self, layer, quarter):
        if (self.isLayerFinished(layer)):
            return True
        layerSlot = self.getLayerSlot(layer)
        xStart, xEnd, yStart, yEnd = self.getQuarterCompletionBounds(layer, quarter)
        return bool(self.pixelHistoryMap[xStart:xEnd, yStart:yEnd, layerSlot].all())

    def getQuarterCompletionBounds(self, layer, quarter):
        xStart, xEnd, yStart, yEnd = self.getQuarterBounds(layer, quarter)
        if (layerList[layer] == "pointwise"):
            # A finished pointwise quarter includes the neighboring row,
            # column, and diagonal corner needed by a 3x3 depthwise kernel.
            spatialSize = int(spatialOutput[layer])
            xStart, xEnd = max(0, xStart - 1), min(spatialSize, xEnd + 1)
            yStart, yEnd = max(0, yStart - 1), min(spatialSize, yEnd + 1)
        return xStart, xEnd, yStart, yEnd

    def getCompletedQuarters(self, layer):
        return tuple(
            quarter for quarter in range(1, SEGMENTS_PER_LAYER + 1)
            if self.isQuarterFinished(layer, quarter))

    def arePixelsFinished(self, layer, pixels):
        if (self.isLayerFinished(layer) or not pixels):
            return True
        layerSlot = self.getLayerSlot(layer)
        x, y = zip(*pixels)
        return bool(self.pixelHistoryMap[list(x), list(y), layerSlot].all())

    def setPixelsFinished(self, layer, pixels):
        if (self.isLayerFinished(layer) or not pixels):
            return
        layerSlot = self.getLayerSlot(layer)
        x, y = zip(*pixels)
        self.pixelHistoryMap[list(x), list(y), layerSlot] = True
        self.advanceCompletedLayers()

    def isRegionFinished(self, layer, xStart, xEnd, yStart, yEnd):
        if (self.isLayerFinished(layer)):
            return True
        layerSpatialSize = spatialOutput[layer]
        if (
                xStart < 0 or xStart > xEnd or xEnd > layerSpatialSize
                or yStart < 0 or yStart > yEnd or yEnd > layerSpatialSize):
            raise Exception("ERROR! Pixel history region is outside the layer bounds")
        layerSlot = self.getLayerSlot(layer)
        return bool(self.pixelHistoryMap[xStart:xEnd, yStart:yEnd, layerSlot].all())

    def isLayerFinished(self, layer):
        if (layer < 0):
            raise Exception("ERROR! Pixel layer cannot be negative")
        return layer <= self.lastCompletedLayer

    def setSEVectorCalculated(self, layer, calculated=True):
        if (layer < 0):
            raise Exception("ERROR! SE vector layer cannot be negative")
        if (calculated):
            self.completedSEVectorLayers.add(layer)
        else:
            self.completedSEVectorLayers.discard(layer)

    def isSEVectorCalculated(self, layer):
        if (layer < 0):
            raise Exception("ERROR! SE vector layer cannot be negative")
        return layer in self.completedSEVectorLayers
    
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
        self.linkedFromAccumulators = []
        self.operationTag = None
        self.operationDSPunits = []
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
        return (
            self.operationTag is None
            and len(self.linkedAccumulators) == 0
            and len(self.linkedFromAccumulators) == 0
            and not self.isActive()
            and not self.ownHasOutput()
        )

    def getOperationTag(self):
        return self.operationTag

    def getOperationDSPunits(self):
        return self.operationDSPunits.copy()

    def getLinkedAccumulatorComponent(self):
        accumulatorComponent = []
        accumulatorsToVisit = [self]
        seenAccumulators = set()

        while (len(accumulatorsToVisit) > 0):
            accumulator = accumulatorsToVisit.pop()
            if (accumulator in seenAccumulators):
                continue

            seenAccumulators.add(accumulator)
            accumulatorComponent.append(accumulator)
            accumulatorsToVisit.extend(accumulator.linkedAccumulators)
            accumulatorsToVisit.extend(accumulator.linkedFromAccumulators)

        return accumulatorComponent

    def setOperationTag(self, operationTag, DSPunits=None):
        if (DSPunits is not None):
            if (any(DSPunit not in self.DSPunits for DSPunit in DSPunits)):
                raise Exception("ERROR! Cannot assign an accumulator operation to a DSP from another accumulator")

        accumulatorComponent = self.getLinkedAccumulatorComponent()
        if (operationTag is not None):
            for accumulator in accumulatorComponent:
                if (accumulator.operationTag is not None and accumulator.operationTag != operationTag):
                    raise Exception("ERROR! Linked DSP accumulators cannot have different operation tags")

        for accumulator in accumulatorComponent:
            accumulator.operationTag = operationTag
            if (operationTag is None):
                accumulator.operationDSPunits.clear()

        if (operationTag is not None and DSPunits is not None):
            for DSPunit in DSPunits:
                if (DSPunit not in self.operationDSPunits):
                    self.operationDSPunits.append(DSPunit)

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
        if (self.operationTag != accumulator.operationTag):
            raise Exception("ERROR! Linked DSP accumulators must have matching operation tags")

        self.linkedAccumulators.append(accumulator)
        accumulator.linkedFromAccumulators.append(self)
        self.linkedOutputReadyClock = None

    def unlinkAccumulator(self, accumulator):
        if (accumulator not in self.linkedAccumulators):
            return

        self.linkedAccumulators.remove(accumulator)
        accumulator.linkedFromAccumulators.remove(self)
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
            return self.accumulatedType in (
                "SE",
                "SEExpansion",
                "FullyConnectedHidden",
                "FullyConnectedOutput") and DSPunit in self.accumulatedDSPunits
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
            self.accumulatedChannels.extend(DSPunit.kernel.inputChannels)

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

    def recordFullyConnectedDotProductStart(self, DSPunit, layer, row, isOutputLayer=False):
        if (DSPunit not in self.DSPunits):
            raise Exception("ERROR! DSP attempted to record fully connected output with the wrong accumulator")
        if (not isinstance(DSPunit.kernel, SEKernel)):
            raise Exception("ERROR! Fully connected accumulator requires a matrix-multiply kernel")

        accumulatedMetadata = (
            self.accumulatedType,
            self.accumulatedLayer,
            self.accumulatedOutputChannel,
            self.accumulatedX,
            self.accumulatedY)
        dspMetadata = (
            "FullyConnectedOutput" if isOutputLayer else "FullyConnectedHidden",
            layer,
            row,
            -1,
            -1)

        if (self.accumulatedType is None):
            (
                self.accumulatedType,
                self.accumulatedLayer,
                self.accumulatedOutputChannel,
                self.accumulatedX,
                self.accumulatedY) = dspMetadata
        elif (accumulatedMetadata != dspMetadata):
            raise Exception("ERROR! DSP accumulator received mismatched fully connected output metadata")

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
        elif (self.accumulatedType in ("FullyConnectedHidden", "FullyConnectedOutput")):
            toReturn = SEVectorIndex(
                type="input" if self.accumulatedType == "FullyConnectedHidden" else "output",
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
            usedAllChannels = accumulatedChannelSet == set(range(channelsIn[self.accumulatedLayer]))

            if (usedAllChannels):
                toReturn = Activation(
                    type="pointwise",
                    layer=self.accumulatedLayer,
                    x=self.accumulatedX // outputStride,
                    y=self.accumulatedY // outputStride,
                    channel=self.accumulatedOutputChannel)
            else:
                toReturn = PartialActivation(
                    type="pointwise",
                    layers=[self.accumulatedLayer],
                    x=self.accumulatedX // outputStride,
                    y=self.accumulatedY // outputStride,
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

        expectedChannel = (self.activationChunk.channel // DSP_CHUNK_SIZE) * DSP_CHUNK_SIZE
        if (self.preAdder.channel != expectedChannel):
            raise Exception("ERROR! Squeeze-excite DSP pre-adder channel does not match activation chunk channel")
        
    def startDotProductUnchecked(self):
        if (isinstance(self.kernel, SEKernel)):
            if (not isinstance(self.preAdder, SEVectorChunk) or self.preAdder.valuesStored <= 0):
                raise Exception("ERROR! SE dot product started without a loaded SE vector chunk")
            if (len(self.kernel.weights) != self.preAdder.valuesStored):
                raise Exception("ERROR! SE dot product weight count does not match SE vector chunk count")
            self.timer = self.preAdder.valuesStored
            self.lastTimerUpdateClockCycle = clkCycles
            return

        if (isinstance(self.kernel, PointwiseKernel)):
            if (not self.kernel.inputChannels):
                raise Exception("ERROR! Pointwise DSP started without any input channels")
            expectedActivationChannel = self.kernel.inputChannels[0]
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
            if (isinstance(self.kernel, DepthwiseKernel)):
                if (self.activationChunk.valuesStored <= 0):
                    raise Exception("ERROR! Depthwise DSP started without any activation values")
                if (self.kernel.valuesStored != self.activationChunk.valuesStored):
                    raise Exception("ERROR! Depthwise kernel and activation chunk sizes do not match")
                if (self.kernel.kernelParts != self.activationChunk.kernelParts):
                    raise Exception("ERROR! Depthwise kernel and activation chunk positions do not match")
                self.timer = self.activationChunk.valuesStored
            else:
                if (self.kernel.inputChannels != self.activationChunk.inputChannels):
                    raise Exception("ERROR! Pointwise kernel and activation input channels do not match")
                if (not 0 < self.activationChunk.valuesStored <= DSP_CHUNK_SIZE
                        or self.activationChunk.valuesStored != len(self.kernel.inputChannels)):
                    raise Exception("ERROR! Pointwise kernel and activation chunk sizes do not match")
                self.validatePreAdderForActivation()
                self.timer = self.activationChunk.valuesStored
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
            # Pointwise handoff is coordinated across the accumulator's DSPs
            # by DSPmanager, after the entire next batch has been preloaded.
            if (self.timer == 0 and not isinstance(self.kernel, PointwiseKernel)):
                self.startPreloadedDotProductIfAvailable()
    
    def isActive(self):
        self.updateTimer()
        return (self.timer > 0)
    
    def getOutput(self):
        raise Exception("ERROR! DSP output must be read from its assigned accumulator")
    
    def setOperationTag(self, operationTag):
        if (self.accumulator is None):
            raise Exception("ERROR! Cannot tag a DSP unit without an accumulator")
        DSPunits = None if operationTag is None else [self]
        self.accumulator.setOperationTag(operationTag, DSPunits)
    
    def getOperationTag(self):
        if (self.accumulator is None):
            return None
        return self.accumulator.getOperationTag()
    
    def getID(self):
        return self.dspID
    
    def getKernel(self):
        return self.kernel

NUM_DSPS = 292
CAMERA_DWC_K = 3
DSPS_PER_ACCUMULATOR = 4
DSP_OPERATION_FIFO_CAPACITY = 50
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
        self.fullyConnectedComplete = False
        
        self.tagCounter = 0
        self.DSPActivationFirstClock = None
        self.DSPActivationLastClock = None
        self.DSPActivationCycleCount = 0
        self.DSPActiveRuns = [[] for _ in range(NUM_DSPS)]
        self.DSPOpenRunStarts = [None] * NUM_DSPS
    
    def getFreeDSPunits(self, max=None):
        if (max is not None and max < 0):
            raise Exception("ERROR! Maximum number of free DSP units cannot be negative")

        freeDSPs = [
            DSPunit
            for accumulator in self.getFreeDSPaccumulators()
            for DSPunit in accumulator.DSPunits
        ]
        if (max is None):
            return freeDSPs
        return freeDSPs[:max]

    def getFreeDSPaccumulators(self, max=None):
        if (max is not None and max < 0):
            raise Exception("ERROR! Maximum number of free DSP accumulators cannot be negative")

        freeAccumulators = [
            accumulator
            for accumulator in self.accumulators
            if accumulator.isFree()
        ]
        if (max is None):
            return freeAccumulators
        return freeAccumulators[:max]

    def getFreeDSPsFromDistinctAccumulators(self, max=None):
        if (max is not None and max < 0):
            raise Exception("ERROR! Maximum number of free DSP units cannot be negative")

        return [
            accumulator.DSPunits[0]
            for accumulator in self.getFreeDSPaccumulators(max)
        ]

    def getFreeAccumulatorDSPGroups(self, max=None):
        if (max is not None and max < 0):
            raise Exception("ERROR! Maximum number of free DSP accumulator groups cannot be negative")

        return [
            accumulator.DSPunits
            for accumulator in self.getFreeDSPaccumulators(max)
        ]

    def linkAccumulator(self, finalAccumulator, linkedAccumulator):
        finalAccumulator.linkAccumulator(linkedAccumulator)

    def unlinkAccumulator(self, finalAccumulator, linkedAccumulator):
        finalAccumulator.unlinkAccumulator(linkedAccumulator)

    def getSEExciteBatchBounds(self, operation: SEVectorProdOperation):
        if (operation.exciteBatchIndex < 0):
            raise Exception("ERROR! SE excite batch index cannot be negative")

        outputChannelCount = channelsOut[operation.layer]
        batchCount = SEExciteOutputChannelDivisor[operation.layer]
        if (batchCount <= 0 or batchCount > outputChannelCount):
            raise Exception("ERROR! SE excite output-channel divisor is invalid")
        if (operation.exciteBatchIndex >= batchCount):
            raise Exception("ERROR! SE excite batch index exceeds the configured divisor")

        channelsPerBatch, largerBatchCount = divmod(outputChannelCount, batchCount)
        batchStart = (
            operation.exciteBatchIndex * channelsPerBatch
            + min(operation.exciteBatchIndex, largerBatchCount))
        batchLength = channelsPerBatch + int(operation.exciteBatchIndex < largerBatchCount)
        return batchStart, batchStart + batchLength

    def getSEOperationOutputChannelBounds(self, operation: SEVectorProdOperation):
        if (operation.isExpansion):
            return self.getSEExciteBatchBounds(operation)
        return 0, SEhiddenLayer[operation.layer]

    def getSEAccumulatorRows(self, operation: SEVectorProdOperation, DSPunits):
        DSPsPerRow = self.getSEDSPsPerRow(
            operation.layer,
            operation.weightsPerDSP,
            operation.isExpansion)
        outputChannelStart, outputChannelEnd = self.getSEOperationOutputChannelBounds(operation)
        outputVectorLength = outputChannelEnd - outputChannelStart

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
        outputChannelStart, _ = self.getSEOperationOutputChannelBounds(operation)
        row = outputChannelStart + (dspIndex // DSPsPerRow)
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

    def getSEDSPsNeededForMatrix(self, layer, weightsPerDSP, isExpansion=False, outputVectorLength=None):
        if (outputVectorLength is None):
            outputVectorLength = self.getSEOutputVectorLength(layer, isExpansion)
        return self.getSEDSPsPerRow(layer, weightsPerDSP, isExpansion) * outputVectorLength

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

    def getFreeDSPsForSERows(self, layer, weightsPerDSP, isExpansion=False, outputVectorLength=None):
        DSPsPerRow = self.getSEDSPsPerRow(layer, weightsPerDSP, isExpansion)
        if (outputVectorLength is None):
            outputVectorLength = self.getSEOutputVectorLength(layer, isExpansion)
        return self.getFreeDSPsForMatrixRows(DSPsPerRow, outputVectorLength)

    def getFreeDSPsForMatrixRows(self, DSPsPerRow, outputVectorLength):
        if (DSPsPerRow <= 0 or outputVectorLength <= 0):
            raise Exception("ERROR! Matrix row dimensions must be positive")
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
        DSPunitsByAccumulator = {}
        for DSPunit in DSPunits:
            if (DSPunit.accumulator is None):
                raise Exception("ERROR! Cannot tag a DSP unit without an accumulator")
            DSPunitsByAccumulator.setdefault(DSPunit.accumulator, []).append(DSPunit)

        for accumulator, accumulatorDSPunits in DSPunitsByAccumulator.items():
            accumulator.setOperationTag(operationTag, accumulatorDSPunits)

    def getDSPsWithOpTag(self, operationTag):
        operationDSPunits = set()
        for accumulator in self.accumulators:
            if (accumulator.getOperationTag() == operationTag):
                operationDSPunits.update(accumulator.getOperationDSPunits())

        return [DSPunit for DSPunit in self.DSPlist if DSPunit in operationDSPunits]
    
    def fifoPush(self, operation: Operation):
        if (len(self.operationFIFO) >= DSP_OPERATION_FIFO_CAPACITY):
            raise Exception(
                f"ERROR! DSP operation FIFO cannot exceed "
                f"{DSP_OPERATION_FIFO_CAPACITY} entries")
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

    def isFullyConnectedCalculating(self):
        return any(isinstance(operation, FullyConnectedOperation) for operation in self.operationFIFO)

    def pwConvEnq(self, layer, quarter):
        return any(
            isinstance(operation, PointwiseSupervisorOperation)
            and operation.layer == layer
            and operation.quarter == quarter
            for operation in self.operationFIFO
        )

    def dwConvEnq(self, layer, quarter):
        return any(
            isinstance(operation, DepthwiseSupervisorOperation)
            and operation.layer == layer
            and operation.quarter == quarter
            for operation in self.operationFIFO
        )

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

            maxWeightsToLoad = DSP_CHUNK_SIZE * 2
            if (hasSEKernelLoaded):
                maxWeightsToLoad = DSP_CHUNK_SIZE

            maxActivationsToLoad = maxWeightsToLoad
            if (not operation.isExpansion and hasSEPreAdderLoaded):
                maxActivationsToLoad = DSP_CHUNK_SIZE

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
        if (not operation.isExpansion):
            squeezeDSPunits = self.getDSPsWithOpTag(operation.tag)
            for DSPunit in squeezeDSPunits:
                DSPunit.setOperationTag(None)

            operation.isExpansion = True
            operation.weightsPerDSP = SEExciteMACsPerDSP[operation.layer]
            operation.exciteBatchIndex = 0
            operation.weightsLoaded = 0
            operation.DSPindexStart = 0
            operation.currentLoadingIndex = 0
            operation.currentUnloadingIndex = 0
            operation.currentOutputStoreIndex = 0
            operation.accumulatorRowsLinked = False
            operation.extractedOutputs.clear()

        outputChannelStart, outputChannelEnd = self.getSEExciteBatchBounds(operation)
        batchOutputChannelCount = outputChannelEnd - outputChannelStart

        freeDSPs = self.getFreeDSPsForSERows(
            operation.layer,
            operation.weightsPerDSP,
            True,
            batchOutputChannelCount)
        expectedDSPCount = self.getSEDSPsNeededForMatrix(
            operation.layer,
            operation.weightsPerDSP,
            True,
            batchOutputChannelCount)
        if (len(freeDSPs) != expectedDSPCount):
            return False

        operation.exciteTag = self.tagCounter
        self.assignOperationTag(freeDSPs, operation.exciteTag)
        self.tagCounter += 1
        operation.exciteOperationStarted = True
        return True

    def completeSEExciteBatch(self, operation: SEVectorProdOperation):
        if (operation.exciteTag is not None):
            for DSPunit in self.getDSPsWithOpTag(operation.exciteTag):
                DSPunit.setOperationTag(None)

        operation.exciteBatchIndex += 1
        if (operation.exciteBatchIndex >= SEExciteOutputChannelDivisor[operation.layer]):
            operation.exciteTag = None
            operation.exciteOperationStarted = False
            return True

        operation.exciteTag = None
        operation.exciteOperationStarted = False
        operation.weightsLoaded = 0
        operation.DSPindexStart = 0
        operation.currentLoadingIndex = 0
        operation.currentUnloadingIndex = 0
        operation.currentOutputStoreIndex = 0
        operation.accumulatorRowsLinked = False
        operation.extractedOutputs.clear()
        return False

    def processSEExciteOperation(self, operation: SEVectorProdOperation):
        if (not operation.exciteOperationStarted):
            if (self.startSEExciteOperation(operation) == False):
                return False

        if (operation.exciteTag is None):
            raise Exception("ERROR! SE excite operation started without an excite tag")

        DSPunits = self.getDSPsWithOpTag(operation.exciteTag)
        outputChannelStart, outputChannelEnd = self.getSEExciteBatchBounds(operation)
        batchOutputChannelCount = outputChannelEnd - outputChannelStart
        expectedDSPCount = self.getSEDSPsNeededForMatrix(
            operation.layer,
            operation.weightsPerDSP,
            True,
            batchOutputChannelCount)
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
            if (not self.extractSEVectorOutputs(operation, DSPunits)):
                return False
            return self.completeSEExciteBatch(operation)

        weightsLoadedThisPass = None
        DSPsPerRow = self.getSEDSPsPerRow(operation.layer, operation.weightsPerDSP, True)
        inputVectorLength = self.getSEInputVectorLength(operation.layer, True)
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

            maxWeightsToLoad = DSP_CHUNK_SIZE * 2
            if (hasSEKernelLoaded):
                maxWeightsToLoad = DSP_CHUNK_SIZE

            maxActivationsToLoad = maxWeightsToLoad
            if (hasSEPreAdderLoaded):
                maxActivationsToLoad = DSP_CHUNK_SIZE

            if (weightsLoadedThisPass is None):
                weightsLoadedThisPass = min(
                    remainingWeightsPerDSP,
                    maxWeightsToLoad,
                    maxActivationsToLoad)

            dspIndexInRow = dspIndex % DSPsPerRow
            dspInputStart = dspIndexInRow * operation.weightsPerDSP
            weightsForDSP = min(
                operation.weightsPerDSP,
                inputVectorLength - dspInputStart)
            remainingWeightsForDSP = weightsForDSP - operation.weightsLoaded
            if (remainingWeightsForDSP <= 0):
                operation.DSPindexStart = dspIndex + 1
                continue

            weightsToLoad = min(
                weightsLoadedThisPass,
                maxWeightsToLoad,
                maxActivationsToLoad,
                remainingWeightsForDSP)
            SEKernels = memory.getSEWeightChunksForDSP(
                operation.layer,
                True,
                dspIndex,
                operation.weightsPerDSP,
                operation.weightsLoaded,
                weightsToLoad,
                outputChannelStart)
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

        if (weightsLoadedThisPass is None):
            raise Exception("ERROR! SE excite operation did not advance its weight load")
        operation.weightsLoaded += weightsLoadedThisPass
        operation.DSPindexStart = 0
        if (operation.weightsLoaded < operation.weightsPerDSP):
            return False
        if (not self.extractSEVectorOutputs(operation, DSPunits)):
            return False
        return self.completeSEExciteBatch(operation)
    
    def getFullyConnectedInputLength(self, stage):
        if (stage == 0):
            return FULLY_CONNECTED_INPUT
        if (stage == 1):
            return FULLY_CONNECTED_HIDDEN
        raise Exception("ERROR! Fully connected stage must be 0 or 1")

    def getFullyConnectedOutputLength(self, stage):
        if (stage == 0):
            return FULLY_CONNECTED_HIDDEN
        if (stage == 1):
            return FULLY_CONNECTED_OUTPUT
        raise Exception("ERROR! Fully connected stage must be 0 or 1")

    def getFullyConnectedLayer(self, stage):
        if (stage < 0 or stage > 1):
            raise Exception("ERROR! Fully connected stage must be 0 or 1")
        return FULLY_CONNECTED_LAYER_START + stage

    def getFullyConnectedBatchBounds(self, operation: FullyConnectedOperation):
        outputLength = self.getFullyConnectedOutputLength(operation.stage)
        batchCount = int(fullyConnectedInputDivisor[operation.stage])
        if (operation.batchIndex < 0 or operation.batchIndex >= batchCount):
            raise Exception("ERROR! Fully connected batch index is outside the configured divisor")

        outputsPerBatch, largerBatchCount = divmod(outputLength, batchCount)
        batchStart = (
            operation.batchIndex * outputsPerBatch
            + min(operation.batchIndex, largerBatchCount))
        batchLength = outputsPerBatch + int(operation.batchIndex < largerBatchCount)
        return batchStart, batchStart + batchLength

    def getFullyConnectedDSPsPerRow(self, operation: FullyConnectedOperation):
        return math.ceil(
            self.getFullyConnectedInputLength(operation.stage)
            / fullyConnectedMACsPerDSP[operation.stage])

    def getFullyConnectedAccumulatorRows(self, operation: FullyConnectedOperation, DSPunits):
        DSPsPerRow = self.getFullyConnectedDSPsPerRow(operation)
        outputStart, outputEnd = self.getFullyConnectedBatchBounds(operation)
        return [
            DSPunits[rowStart:rowStart + DSPsPerRow]
            for rowStart in range(0, DSPsPerRow * (outputEnd - outputStart), DSPsPerRow)
        ]

    def linkFullyConnectedAccumulatorRows(self, operation: FullyConnectedOperation, DSPunits):
        for rowDSPs in self.getFullyConnectedAccumulatorRows(operation, DSPunits):
            finalAccumulator = rowDSPs[0].accumulator
            linkedAccumulators = set()
            for DSPunit in rowDSPs[1:]:
                if (DSPunit.accumulator is finalAccumulator or DSPunit.accumulator in linkedAccumulators):
                    continue
                finalAccumulator.linkAccumulator(DSPunit.accumulator)
                linkedAccumulators.add(DSPunit.accumulator)
        operation.accumulatorRowsLinked = True

    def recordFullyConnectedAccumulatorStart(
            self,
            DSPunit,
            operation: FullyConnectedOperation,
            dspIndex):
        DSPsPerRow = self.getFullyConnectedDSPsPerRow(operation)
        outputStart, _ = self.getFullyConnectedBatchBounds(operation)
        outputChannel = outputStart + (dspIndex // DSPsPerRow)
        DSPunit.accumulator.recordFullyConnectedDotProductStart(
            DSPunit,
            self.getFullyConnectedLayer(operation.stage),
            outputChannel,
            operation.stage == 1)

    def resetFullyConnectedBatchState(self, operation: FullyConnectedOperation):
        operation.batchStarted = False
        operation.batchTag = None
        operation.weightsLoaded = 0
        operation.DSPindexStart = 0
        operation.currentUnloadingIndex = 0
        operation.currentOutputStoreIndex = 0
        operation.accumulatorRowsLinked = False
        operation.extractedOutputs.clear()

    def startFullyConnectedBatch(self, operation: FullyConnectedOperation):
        outputStart, outputEnd = self.getFullyConnectedBatchBounds(operation)
        DSPsPerRow = self.getFullyConnectedDSPsPerRow(operation)
        freeDSPs = self.getFreeDSPsForMatrixRows(DSPsPerRow, outputEnd - outputStart)
        expectedDSPCount = DSPsPerRow * (outputEnd - outputStart)
        if (len(freeDSPs) != expectedDSPCount):
            return False

        operation.batchTag = self.tagCounter
        self.assignOperationTag(freeDSPs, operation.batchTag)
        self.tagCounter += 1
        operation.batchStarted = True
        return True

    def extractFullyConnectedOutputs(self, operation: FullyConnectedOperation, DSPunits):
        rowDSPsList = self.getFullyConnectedAccumulatorRows(operation, DSPunits)
        expectedLayer = self.getFullyConnectedLayer(operation.stage)

        while (operation.currentUnloadingIndex < len(rowDSPsList)):
            rowDSPs = rowDSPsList[operation.currentUnloadingIndex]
            finalAccumulator = rowDSPs[0].accumulator
            if (not finalAccumulator.hasOutputReady()):
                break

            output = finalAccumulator.getOutput()
            expectedType = "input" if operation.stage == 0 else "output"
            if (
                    not isinstance(output, SEVectorIndex)
                    or output.type != expectedType
                    or output.layer != expectedLayer):
                raise Exception("ERROR! Fully connected accumulator produced the wrong output")
            operation.extractedOutputs.append(output)
            finalAccumulator.clearOutput()
            self.unlinkSEAccumulatorRow(rowDSPs)
            operation.currentUnloadingIndex += 1

        if (operation.currentUnloadingIndex < len(rowDSPsList)):
            return False

        while (operation.currentOutputStoreIndex < len(operation.extractedOutputs)):
            output = operation.extractedOutputs[operation.currentOutputStoreIndex]
            if (not memory.loadFullyConnectedVectorPart(output)):
                return False
            operation.currentOutputStoreIndex += 1
        return True

    def completeFullyConnectedBatch(self, operation: FullyConnectedOperation):
        if (operation.batchTag is not None):
            for DSPunit in self.getDSPsWithOpTag(operation.batchTag):
                DSPunit.setOperationTag(None)

        operation.batchIndex += 1
        batchCount = int(fullyConnectedInputDivisor[operation.stage])
        if (operation.batchIndex < batchCount):
            self.resetFullyConnectedBatchState(operation)
            return False

        if (operation.stage == 0):
            # Every GAP average has now been consumed and the entire hidden
            # vector has been committed to SRAM for the output stage.
            memory.clearAccumulator()
            operation.stage = 1
            operation.batchIndex = 0
            self.resetFullyConnectedBatchState(operation)
            return False

        operation.batchTag = None
        operation.batchStarted = False
        return True

    def processFullyConnectedOperation(self, operation: FullyConnectedOperation):
        if (not operation.batchStarted):
            if (not self.startFullyConnectedBatch(operation)):
                return False

        if (operation.batchTag is None):
            raise Exception("ERROR! Fully connected batch started without an operation tag")

        DSPunits = self.getDSPsWithOpTag(operation.batchTag)
        outputStart, outputEnd = self.getFullyConnectedBatchBounds(operation)
        DSPsPerRow = self.getFullyConnectedDSPsPerRow(operation)
        expectedDSPCount = DSPsPerRow * (outputEnd - outputStart)
        if (len(DSPunits) != expectedDSPCount):
            raise Exception("ERROR! Fully connected batch did not reserve its complete DSP matrix")

        weightsPerDSP = int(fullyConnectedMACsPerDSP[operation.stage])
        if (operation.weightsLoaded < 0 or operation.weightsLoaded > weightsPerDSP):
            raise Exception("ERROR! Fully connected weight cursor is outside the DSP allocation")
        if (operation.DSPindexStart < 0 or operation.DSPindexStart > len(DSPunits)):
            raise Exception("ERROR! Fully connected DSP cursor is outside the allocated matrix")

        if (not operation.accumulatorRowsLinked):
            self.linkFullyConnectedAccumulatorRows(operation, DSPunits)

        remainingWeightsPerDSP = weightsPerDSP - operation.weightsLoaded
        if (remainingWeightsPerDSP == 0):
            if (not self.extractFullyConnectedOutputs(operation, DSPunits)):
                return False
            return self.completeFullyConnectedBatch(operation)

        weightsLoadedThisPass = None
        for dspIndex in range(operation.DSPindexStart, len(DSPunits)):
            DSPunit = DSPunits[dspIndex]
            DSPunit.isActive()
            hasKernelLoaded = isinstance(DSPunit.kernel, SEKernel)
            hasKernelPreloaded = isinstance(DSPunit.preloadedKernel, SEKernel)
            hasInputLoaded = isinstance(DSPunit.preAdder, SEVectorChunk) and DSPunit.preAdder.valuesStored > 0
            hasInputPreloaded = (
                isinstance(DSPunit.preloadedPreAdder, SEVectorChunk)
                and DSPunit.preloadedPreAdder.valuesStored > 0)

            if ((not DSPunit.isActive()) and hasKernelLoaded and (not hasKernelPreloaded)):
                DSPunit.clearActiveStorage()
                hasKernelLoaded = False
                hasInputLoaded = False

            if ((not DSPunit.isActive()) and DSPunit.hasPreloadedDotProduct()):
                DSPunit.startPreloadedDotProductIfAvailable()
                self.recordFullyConnectedAccumulatorStart(DSPunit, operation, dspIndex)
                hasKernelLoaded = isinstance(DSPunit.kernel, SEKernel)
                hasKernelPreloaded = isinstance(DSPunit.preloadedKernel, SEKernel)
                hasInputLoaded = isinstance(DSPunit.preAdder, SEVectorChunk) and DSPunit.preAdder.valuesStored > 0
                hasInputPreloaded = (
                    isinstance(DSPunit.preloadedPreAdder, SEVectorChunk)
                    and DSPunit.preloadedPreAdder.valuesStored > 0)

            if ((not hasKernelLoaded) and hasKernelPreloaded):
                DSPunit.loadKernel()
                hasKernelLoaded = True
                hasKernelPreloaded = False
            if ((not hasInputLoaded) and hasInputPreloaded):
                DSPunit.loadPreAdder()
                hasInputLoaded = True
                hasInputPreloaded = False

            if (hasKernelLoaded and hasKernelPreloaded):
                return False
            if (hasInputLoaded and hasInputPreloaded):
                return False

            maxWeightsToLoad = DSP_CHUNK_SIZE * 2
            if (hasKernelLoaded):
                maxWeightsToLoad = DSP_CHUNK_SIZE
            maxInputsToLoad = maxWeightsToLoad
            if (hasInputLoaded):
                maxInputsToLoad = DSP_CHUNK_SIZE

            if (weightsLoadedThisPass is None):
                weightsLoadedThisPass = min(
                    remainingWeightsPerDSP,
                    maxWeightsToLoad,
                    maxInputsToLoad)
            valuesToLoad = min(weightsLoadedThisPass, maxWeightsToLoad, maxInputsToLoad)

            kernels = memory.getFullyConnectedWeightChunksForDSP(
                operation.stage,
                dspIndex,
                weightsPerDSP,
                operation.weightsLoaded,
                valuesToLoad,
                outputStart)
            if (kernels == False):
                operation.DSPindexStart = dspIndex
                return False

            inputChunks = memory.getFullyConnectedInputChunksForDSP(
                operation.stage,
                dspIndex,
                weightsPerDSP,
                operation.weightsLoaded,
                valuesToLoad)
            if (inputChunks == False):
                operation.DSPindexStart = dspIndex
                return False

            if (hasKernelLoaded):
                DSPunit.preloadKernel(kernels[0])
                DSPunit.preloadPreAdder(inputChunks[0])
            else:
                DSPunit.preloadKernel(kernels[0])
                DSPunit.preloadPreAdder(inputChunks[0])
                DSPunit.loadKernel()
                DSPunit.loadPreAdder()
                DSPunit.startDotProduct()
                self.recordFullyConnectedAccumulatorStart(DSPunit, operation, dspIndex)
                if (len(kernels) > 1):
                    DSPunit.preloadKernel(kernels[1])
                    DSPunit.preloadPreAdder(inputChunks[1])

            operation.DSPindexStart = dspIndex + 1

        if (weightsLoadedThisPass is None):
            raise Exception("ERROR! Fully connected operation did not advance its weight load")
        operation.weightsLoaded += weightsLoadedThisPass
        operation.DSPindexStart = 0
        if (operation.weightsLoaded < weightsPerDSP):
            return False
        if (not self.extractFullyConnectedOutputs(operation, DSPunits)):
            return False
        return self.completeFullyConnectedBatch(operation)

    def startDepthwiseConvFromCamera(self, targetX, targetY, camera: camera):
        kernelRadius = DEPTHWISE_KERNEL_SIZE // 2
        lastNeededX = min(targetX + kernelRadius, camera.width - 1)
        lastNeededY = min(targetY + kernelRadius, camera.height - 1)
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

    def getPointwiseSegmentPixels(self, layer, quarter, quarterSegment):
        if (layer < 0 or layer >= layers):
            raise Exception("ERROR! Pointwise layer is outside the layer range")
        if (layerList[layer] != "pointwise"):
            raise Exception("ERROR! Pointwise operation requested for a non-pointwise layer")
        if (quarter < 1 or quarter > SEGMENTS_PER_LAYER):
            raise Exception("ERROR! Quarter must be 1, 2, 3, or 4")

        division = int(pointwiseQuarterDivision[layer])
        if (division <= 0 or (division & (division - 1)) != 0):
            raise Exception("ERROR! Pointwise quarter division must be a positive power of two")
        if (quarterSegment < 0 or quarterSegment >= division):
            raise Exception("ERROR! Pointwise quarter segment is outside the quarter division")

        spatialSize = int(spatialOutput[layer])
        halfSpatialSize = spatialSize // 2
        if (spatialSize % 2 != 0):
            raise Exception("ERROR! Pointwise spatial size must be divisible into four equal quarters")

        quarterColumn = (quarter - 1) % 2
        quarterRow = (quarter - 1) // 2
        quarterXStart = quarterColumn * halfSpatialSize
        quarterYStart = quarterRow * halfSpatialSize

        # Keep power-of-two subdivisions as square as possible. For example,
        # division 16 becomes a 4x4 row-major grid, matching quarterSegment 0-15.
        divisionExponent = division.bit_length() - 1
        segmentColumns = 1 << ((divisionExponent + 1) // 2)
        segmentRows = division // segmentColumns
        if (halfSpatialSize % segmentColumns != 0 or halfSpatialSize % segmentRows != 0):
            raise Exception("ERROR! Pointwise quarter cannot be evenly divided into the requested grid")

        segmentWidth = halfSpatialSize // segmentColumns
        segmentHeight = halfSpatialSize // segmentRows
        segmentColumn = quarterSegment % segmentColumns
        segmentRow = quarterSegment // segmentColumns
        xStart = quarterXStart + segmentColumn * segmentWidth
        yStart = quarterYStart + segmentRow * segmentHeight

        return [
            (x, y)
            for y in range(yStart, yStart + segmentHeight)
            for x in range(xStart, xStart + segmentWidth)
        ]

    def getPointwiseQuarterSegments(self, layer, quarter, coveredQuarters):
        coveredBounds = [
            pixelMap.getQuarterCompletionBounds(layer, coveredQuarter)
            for coveredQuarter in coveredQuarters]

        def needsCompute(x, y):
            return not any(
                xStart <= x < xEnd and yStart <= y < yEnd
                for xStart, xEnd, yStart, yEnd in coveredBounds)

        # Preserve the spreadsheet's core tile order and add the outside border
        # afterward. Split both into batches that fit the physical DSP array.
        coreSegments = [
            self.getPointwiseSegmentPixels(layer, quarter, segment)
            for segment in range(int(pointwiseQuarterDivision[layer]))]
        batchCapacity = min(len(coreSegments[0]), len(self.accumulators))
        xStart, xEnd, yStart, yEnd = pixelMap.getQuarterCompletionBounds(layer, quarter)
        coreXStart, coreXEnd, coreYStart, coreYEnd = pixelMap.getQuarterBounds(layer, quarter)
        borderPixels = [
            (x, y)
            for y in range(yStart, yEnd)
            for x in range(xStart, xEnd)
            if not (coreXStart <= x < coreXEnd and coreYStart <= y < coreYEnd)]
        segments = []
        for pixels in coreSegments + [borderPixels]:
            remainingPixels = [pixel for pixel in pixels if needsCompute(*pixel)]
            for offset in range(0, len(remainingPixels), batchCapacity):
                segments.append(tuple(remainingPixels[offset:offset + batchCapacity]))
        return tuple(segments)

    def getPointwiseSRAMStoreOrder(self, layer, segmentPixels):
        scheme = str(pointwiseSRAMScheme[layer]).strip()
        if (scheme == "basic"):
            return list(range(len(segmentPixels)))

        if (len(segmentPixels) == 0):
            return []

        minX = min(x for x, _ in segmentPixels)
        minY = min(y for _, y in segmentPixels)

        if (scheme == "altCol"):
            return [
                index
                for columnParity in range(2)
                for index, (x, _) in enumerate(segmentPixels)
                if ((x - minX) % 2 == columnParity)
            ]

        if (scheme == "altRow"):
            return [
                index
                for rowParity in range(2)
                for index, (_, y) in enumerate(segmentPixels)
                if ((y - minY) % 2 == rowParity)
            ]

        raise Exception(
            f'ERROR! Unknown pointwise SRAM loading scheme "{scheme}" for layer {layer}')

    def pointwiseKernelMatches(self, kernel, layer, outputChannel, inputChannels):
        return (
            isinstance(kernel, PointwiseKernel)
            and kernel.layer == layer
            and kernel.channel == outputChannel
            and kernel.inputChannels == inputChannels)

    def pointwiseActivationMatches(self, activationChunk, layer, x, y, inputChannels):
        return (
            isinstance(activationChunk, ActivationChunk)
            and activationChunk.type == "pointwise"
            and activationChunk.layer == layer
            and activationChunk.x == x
            and activationChunk.y == y
            and activationChunk.inputChannels == inputChannels
            and activationChunk.valuesStored == len(inputChannels))

    def getPointwiseSupervisorByTag(self, supervisorTag):
        matchingSupervisors = [
            operation
            for operation in self.operationFIFO
            if (
                isinstance(operation, PointwiseSupervisorOperation)
                and operation.tag == supervisorTag)
        ]
        if (len(matchingSupervisors) != 1):
            raise Exception("ERROR! Pointwise child operation could not find its supervisor")
        return matchingSupervisors[0]

    def notifyPointwiseChildComplete(self, operation: PointwiseConvOperation):
        supervisor = self.getPointwiseSupervisorByTag(operation.supervisorTag)
        if (
                supervisor.layer != operation.layer
                or supervisor.quarter != operation.quarter
                or supervisor.quarterSegment != operation.quarterSegment):
            raise Exception("ERROR! Pointwise child completion does not match its supervisor")
        if (supervisor.childrenInFlight <= 0):
            raise Exception("ERROR! Pointwise supervisor received an unexpected child completion")
        if (supervisor.childrenCompleted >= supervisor.nextOutputChannel):
            raise Exception("ERROR! Pointwise supervisor received too many child completions")

        supervisor.childrenInFlight -= 1
        supervisor.childrenCompleted += 1

    def startPointwiseChildOperation(
            self,
            supervisor: PointwiseSupervisorOperation,
            outputChannel):
        segmentPixels = supervisor.pixelSegments[supervisor.quarterSegment]
        # Border inputs may belong to source quarters that are still running.
        # Wait before reserving DSPs or trying to read their activations.
        inputPixels = tuple(
            (x * stride[supervisor.layer], y * stride[supervisor.layer])
            for x, y in segmentPixels)
        if (not pixelMap.arePixelsFinished(supervisor.layer - 1, inputPixels)):
            return False
        freeDSPGroups = self.getFreeAccumulatorDSPGroups(len(segmentPixels))
        if (len(freeDSPGroups) != len(segmentPixels)):
            return False

        freeDSPs = [
            DSPunit
            for DSPGroup in freeDSPGroups
            for DSPunit in DSPGroup
        ]
        childTag = self.tagCounter
        self.assignOperationTag(freeDSPs, childTag)
        pointwiseConvOp = PointwiseConvOperation(
            layer=supervisor.layer,
            tag=childTag,
            supervisorTag=supervisor.tag,
            quarter=supervisor.quarter,
            quarterSegment=supervisor.quarterSegment,
            outputChannel=outputChannel,
            groupStates=[PointwiseGroupState() for _ in freeDSPGroups],
            pixels=segmentPixels)
        self.tagCounter += 1
        self.fifoPush(pointwiseConvOp)
        return True

    def processPointwiseSupervisorOperation(
            self,
            operation: PointwiseSupervisorOperation):
        if (operation.quarterSegment >= len(operation.pixelSegments)):
            # Work owned by an earlier queued quarter can still be in flight.
            # Completing our own pixels alone does not guarantee the border.
            return pixelMap.isQuarterFinished(operation.layer, operation.quarter)
        if (operation.childrenInFlight < 0 or operation.childrenCompleted < 0):
            raise Exception("ERROR! Pointwise supervisor child counters cannot be negative")
        if (operation.nextOutputChannel > channelsOut[operation.layer]):
            raise Exception("ERROR! Pointwise supervisor issued too many output channels")
        if (operation.childrenCompleted + operation.childrenInFlight != operation.nextOutputChannel):
            raise Exception("ERROR! Pointwise supervisor child counters are inconsistent")

        allChildrenIssued = operation.nextOutputChannel == channelsOut[operation.layer]
        if (allChildrenIssued and operation.childrenInFlight == 0):
            if (operation.childrenCompleted != channelsOut[operation.layer]):
                raise Exception("ERROR! Pointwise supervisor completed without all output channels")

            pixelMap.setPixelsFinished(
                operation.layer, operation.pixelSegments[operation.quarterSegment])
            operation.quarterSegment += 1
            if (operation.quarterSegment >= len(operation.pixelSegments)):
                return pixelMap.isQuarterFinished(operation.layer, operation.quarter)

            operation.nextOutputChannel = 0
            operation.childrenCompleted = 0

        # Keep issuing children until the next output channel cannot reserve a
        # complete accumulator group for every pixel in the current segment.
        while (operation.nextOutputChannel < channelsOut[operation.layer]):
            if (not self.startPointwiseChildOperation(
                    operation,
                    operation.nextOutputChannel)):
                break

            operation.nextOutputChannel += 1
            operation.childrenInFlight += 1

        return False

    def preloadPointwiseGroup(
            self,
            operation: PointwiseConvOperation,
            DSPGroup,
            outputX,
            outputY,
            firstChannel,
            channelsThisBatch):
        inputX = outputX * stride[operation.layer]
        inputY = outputY * stride[operation.layer]
        inputChannelsByDSP = [
            tuple(range(
                firstChannel + DSPoffset,
                firstChannel + channelsThisBatch,
                DSPS_PER_ACCUMULATOR))
            for DSPoffset in range(min(channelsThisBatch, DSPS_PER_ACCUMULATOR))
        ]
        participatingDSPs = DSPGroup[:len(inputChannelsByDSP)]

        for DSPunit, inputChannels in zip(participatingDSPs, inputChannelsByDSP):
            if (not self.pointwiseKernelMatches(
                    DSPunit.preloadedKernel,
                    operation.layer,
                    operation.outputChannel,
                    inputChannels)):
                desiredKernel = PointwiseKernel(
                    layer=operation.layer,
                    channel=operation.outputChannel,
                    inputChannels=inputChannels)
                if (not memory.askForKernel(desiredKernel, DSPunit.getID())):
                    continue

            if (not self.pointwiseActivationMatches(
                    DSPunit.preloadedActivationChunk,
                    operation.layer,
                    inputX,
                    inputY,
                    inputChannels)):
                memory.askForPointwiseActivationChunk(
                    operation.layer,
                    inputX,
                    inputY,
                    inputChannels,
                    DSPunit.getID())

        # Only this pixel's four-DSP group must be ready. Other pixels can
        # continue loading or computing independently.
        if (not all(
                self.pointwiseKernelMatches(
                    DSPunit.preloadedKernel,
                    operation.layer,
                    operation.outputChannel,
                    inputChannels)
                and self.pointwiseActivationMatches(
                    DSPunit.preloadedActivationChunk,
                    operation.layer,
                    inputX,
                    inputY,
                    inputChannels)
                for DSPunit, inputChannels in zip(participatingDSPs, inputChannelsByDSP))):
            return False

        return True

    def processPointwiseConvOperation(self, operation: PointwiseConvOperation):
        DSPunits = self.getDSPsWithOpTag(operation.tag)
        DSPGroups = [
            DSPunits[index:index + DSPS_PER_ACCUMULATOR]
            for index in range(0, len(DSPunits), DSPS_PER_ACCUMULATOR)
        ]
        segmentPixels = operation.pixels
        if (segmentPixels is None):
            segmentPixels = self.getPointwiseSegmentPixels(
                operation.layer,
                operation.quarter,
                operation.quarterSegment)

        if (
                len(DSPGroups) != len(segmentPixels)
                or len(operation.groupStates) != len(DSPGroups)
                or any(len(DSPGroup) != DSPS_PER_ACCUMULATOR for DSPGroup in DSPGroups)):
            raise Exception("ERROR! Pointwise operation requires one DSP accumulator group per pixel")

        # Each accumulator has an active batch and one preload batch, each
        # containing up to eight pairs total (two per DSP). Fill the next
        # batch while the current batch computes, then hand it off as a group.
        for DSPGroup, state, (outputX, outputY) in zip(
                DSPGroups, operation.groupStates, segmentPixels):
            if (state.outputStored):
                continue
            if (state.channelsInFlight > 0):
                if (not any(DSPunit.isActive() for DSPunit in DSPGroup)):
                    state.channelsCompleted += state.channelsInFlight
                    state.channelsInFlight = 0

            firstChannelToPreload = state.channelsCompleted + state.channelsInFlight
            remainingChannels = int(channelsIn[operation.layer] - firstChannelToPreload)
            if (remainingChannels == 0):
                continue
            channelsThisBatch = min(
                remainingChannels,
                DSPS_PER_ACCUMULATOR * POINTWISE_VALUES_PER_DSP)
            nextBatchReady = self.preloadPointwiseGroup(
                    operation,
                    DSPGroup,
                    outputX,
                    outputY,
                    firstChannelToPreload,
                    channelsThisBatch)
            if (state.channelsInFlight == 0 and nextBatchReady):
                for DSPunit in DSPGroup[:min(channelsThisBatch, DSPS_PER_ACCUMULATOR)]:
                    DSPunit.loadKernel()
                    DSPunit.loadActivation()
                    DSPunit.startDotProduct()
                state.channelsInFlight = channelsThisBatch

        # Retire ready pixels independently. A busy SRAM or an unfinished
        # group does not prevent another pixel from storing its output.
        SRAMStoreOrder = self.getPointwiseSRAMStoreOrder(operation.layer, segmentPixels)
        for DSPGroupIndex in SRAMStoreOrder:
            state = operation.groupStates[DSPGroupIndex]
            if (state.outputStored or state.channelsCompleted != channelsIn[operation.layer]):
                continue
            accumulator = DSPGroups[DSPGroupIndex][0].accumulator
            if (not accumulator.hasOutputReady()):
                continue

            activation = accumulator.getOutput()
            if (not isinstance(activation, Activation)):
                raise Exception("ERROR! Completed pointwise accumulation did not produce an activation")
            if (activation.channel != operation.outputChannel):
                raise Exception("ERROR! Pointwise output channel does not match its operation")
            expectedX, expectedY = segmentPixels[DSPGroupIndex]
            if (activation.x != expectedX or activation.y != expectedY):
                raise Exception("ERROR! Pointwise output pixel does not match its DSP accumulator group")
            if (not memory.loadActivationToSRAM(activation)):
                continue

            accumulator.clearOutput()
            state.outputStored = True

        return all(state.outputStored for state in operation.groupStates)

    def startPointwiseConv(self, layer, quarter, completedQuarters=None):
        self.getPointwiseSegmentPixels(layer, quarter, 0)
        if (not pixelMap.isQuarterFinished(layer - 1, quarter)):
            raise Exception("Attempting to pointwise convolution a non-existent layer quarter")
        if (pixelMap.isQuarterFinished(layer, quarter)):
            return False
        if (self.pwConvEnq(layer, quarter)):
            return False

        if (completedQuarters is None):
            completedQuarters = pixelMap.getCompletedQuarters(layer)
        completedQuarters = tuple(completedQuarters)
        for completedQuarter in completedQuarters:
            if (not pixelMap.isQuarterFinished(layer, completedQuarter)):
                raise Exception("ERROR! Pointwise enqueue context includes an unfinished quarter")
        pendingQuarters = tuple(
            operation.quarter for operation in self.operationFIFO
            if isinstance(operation, PointwiseSupervisorOperation) and operation.layer == layer)
        # Earlier supervisors own overlapping pixels until their stores finish;
        # later supervisors skip those pixels and wait for their completion.
        pixelSegments = self.getPointwiseQuarterSegments(
            layer, quarter, completedQuarters + pendingQuarters)
        pointwiseSupervisorOp = PointwiseSupervisorOperation(
            layer=layer,
            tag=self.tagCounter,
            quarter=quarter,
            completedQuarters=completedQuarters,
            pendingQuarters=pendingQuarters,
            pixelSegments=pixelSegments)
        self.tagCounter += 1
        self.fifoPush(pointwiseSupervisorOp)
        return True

    def getDepthwiseQuarterPixels(self, layer, quarter):
        xStart, xEnd, yStart, yEnd = pixelMap.getQuarterBounds(layer, quarter)
        return [
            (x, y)
            for y in range(yStart, yEnd)
            for x in range(xStart, xEnd)
        ]

    def isDepthwiseInputRegionFinished(self, layer, quarter):
        outputXStart, outputXEnd, outputYStart, outputYEnd = (
            pixelMap.getQuarterBounds(layer, quarter)
        )
        sourceLayer = layer - 1
        kernelRadius = DEPTHWISE_KERNEL_SIZE // 2
        inputXStart = max(0, outputXStart * stride[layer] - kernelRadius)
        inputXEnd = min(
            spatialOutput[sourceLayer],
            (outputXEnd - 1) * stride[layer] + kernelRadius + 1)
        inputYStart = max(0, outputYStart * stride[layer] - kernelRadius)
        inputYEnd = min(
            spatialOutput[sourceLayer],
            (outputYEnd - 1) * stride[layer] + kernelRadius + 1)
        return pixelMap.isRegionFinished(
            sourceLayer,
            inputXStart,
            inputXEnd,
            inputYStart,
            inputYEnd)

    def getDepthwiseDSPKernelParts(self, layer, outputX, outputY):
        inputCenterX = outputX * stride[layer]
        inputCenterY = outputY * stride[layer]
        inputSpatialSize = spatialOutput[layer - 1]
        kernelRadius = DEPTHWISE_KERNEL_SIZE // 2
        staticParts = ((0, 1, 2), (3, 4), (5, 6), (7, 8))
        return [
            tuple(
                kernelPart
                for kernelPart in DSPParts
                if (
                    0 <= inputCenterX + (kernelPart % DEPTHWISE_KERNEL_SIZE) - kernelRadius < inputSpatialSize
                    and 0 <= inputCenterY + (kernelPart // DEPTHWISE_KERNEL_SIZE) - kernelRadius < inputSpatialSize)
            )
            for DSPParts in staticParts
        ]

    def depthwiseKernelMatches(self, kernel, layer, channel, kernelParts):
        return (
            isinstance(kernel, DepthwiseKernel)
            and kernel.layer == layer
            and kernel.channel == channel
            and kernel.kernelParts == tuple(kernelParts)
            and kernel.valuesStored == len(kernelParts))

    def depthwiseActivationMatches(
            self,
            activationChunk,
            layer,
            outputX,
            outputY,
            channel,
            kernelParts):
        return (
            isinstance(activationChunk, ActivationChunk)
            and activationChunk.type == "depthwise"
            and activationChunk.layer == layer
            and activationChunk.x == outputX * stride[layer]
            and activationChunk.y == outputY * stride[layer]
            and activationChunk.channel == channel
            and activationChunk.kernelParts == tuple(kernelParts)
            and activationChunk.valuesStored == len(kernelParts))

    def activationAskForKernel(
            self,
            operation: DepthwiseConvOperation,
            DSPGroups,
            batchPixels):
        # The full nine-weight kernel is fetched from LRAM once per channel
        # child and retained in a local operation buffer.
        if (operation.bufferedKernel is None):
            kernelLoadDSP = DSPGroups[0][0]
            desiredKernel = DepthwiseKernel(
                layer=operation.layer,
                channel=operation.channel)
            if (not memory.askForKernel(desiredKernel, kernelLoadDSP.getID())):
                return False
            if (kernelLoadDSP.preloadedKernel != desiredKernel):
                raise Exception("ERROR! Depthwise kernel buffer received the wrong complete kernel")
            operation.bufferedKernel = replace(kernelLoadDSP.preloadedKernel)

        # Split and broadcast the buffered kernel into the four DSP preload
        # slots for every pixel. getDepthwiseDSPKernelParts removes positions
        # whose activations are outside the frame.
        if (not operation.kernelDistributedForBatch):
            completeKernelParts = set(operation.bufferedKernel.kernelParts)
            if (completeKernelParts != set(range(DEPTHWISE_KERNEL_PARTS))):
                raise Exception("ERROR! Buffered depthwise kernel is incomplete")

            for groupIndex, (outputX, outputY) in enumerate(batchPixels):
                kernelPartsByDSP = self.getDepthwiseDSPKernelParts(
                    operation.layer,
                    outputX,
                    outputY)
                for DSPoffset, kernelParts in enumerate(kernelPartsByDSP):
                    DSPunit = DSPGroups[groupIndex][DSPoffset]
                    if (len(kernelParts) == 0):
                        DSPunit.preloadedKernel = DepthwiseKernel()
                        DSPunit.preloadedActivationChunk = ActivationChunk(type="depthwise")
                        continue
                    if (not set(kernelParts).issubset(completeKernelParts)):
                        raise Exception("ERROR! Depthwise kernel split requested an unavailable weight")
                    DSPunit.preloadKernel(DepthwiseKernel(
                        layer=operation.layer,
                        channel=operation.channel,
                        kernelParts=tuple(kernelParts),
                        valuesStored=len(kernelParts)))

            operation.kernelDistributedForBatch = True

        # Load the activation positions paired with each DSP's kernel slots.
        # The cursor makes SRAM conflicts resumable without rereading LRAM.
        totalDSPsToLoad = len(batchPixels) * DSPS_PER_ACCUMULATOR
        while (operation.nextDSPToLoad < totalDSPsToLoad):
            groupIndex = operation.nextDSPToLoad // DSPS_PER_ACCUMULATOR
            DSPoffset = operation.nextDSPToLoad % DSPS_PER_ACCUMULATOR
            outputX, outputY = batchPixels[groupIndex]
            kernelParts = self.getDepthwiseDSPKernelParts(
                operation.layer,
                outputX,
                outputY)[DSPoffset]
            DSPunit = DSPGroups[groupIndex][DSPoffset]

            if (len(kernelParts) == 0):
                operation.nextDSPToLoad += 1
                continue
            if (not self.depthwiseKernelMatches(
                    DSPunit.preloadedKernel,
                    operation.layer,
                    operation.channel,
                    kernelParts)):
                raise Exception("ERROR! Depthwise kernel was split into the wrong DSP slot")
            if (not self.depthwiseActivationMatches(
                    DSPunit.preloadedActivationChunk,
                    operation.layer,
                    outputX,
                    outputY,
                    operation.channel,
                    kernelParts)):
                if (not memory.askForDepthwiseActivationChunk(
                        operation.layer,
                        outputX,
                        outputY,
                        operation.channel,
                        kernelParts,
                        DSPunit.getID())):
                    return False

            operation.nextDSPToLoad += 1

        return True

    def getDepthwiseSupervisorByTag(self, supervisorTag):
        matchingSupervisors = [
            operation
            for operation in self.operationFIFO
            if (
                isinstance(operation, DepthwiseSupervisorOperation)
                and operation.tag == supervisorTag)
        ]
        if (len(matchingSupervisors) != 1):
            raise Exception("ERROR! Depthwise child operation could not find its supervisor")
        return matchingSupervisors[0]

    def notifyDepthwiseChildComplete(self, operation: DepthwiseConvOperation):
        supervisor = self.getDepthwiseSupervisorByTag(operation.supervisorTag)
        if (supervisor.layer != operation.layer or supervisor.quarter != operation.quarter):
            raise Exception("ERROR! Depthwise child completion does not match its supervisor")
        if (supervisor.childrenInFlight <= 0):
            raise Exception("ERROR! Depthwise supervisor received an unexpected child completion")
        supervisor.childrenInFlight -= 1
        supervisor.childrenCompleted += 1

    def startDepthwiseChildOperation(
            self,
            supervisor: DepthwiseSupervisorOperation,
            channel):
        quarterPixels = self.getDepthwiseQuarterPixels(
            supervisor.layer,
            supervisor.quarter)
        freeDSPGroups = self.getFreeAccumulatorDSPGroups(len(quarterPixels))
        if (len(freeDSPGroups) == 0):
            return False

        freeDSPs = [
            DSPunit
            for DSPGroup in freeDSPGroups
            for DSPunit in DSPGroup
        ]
        childTag = self.tagCounter
        self.assignOperationTag(freeDSPs, childTag)
        self.fifoPush(DepthwiseConvOperation(
            layer=supervisor.layer,
            tag=childTag,
            supervisorTag=supervisor.tag,
            stride=int(stride[supervisor.layer]),
            quarter=supervisor.quarter,
            channel=channel,
            numDSPGroups=len(freeDSPGroups)))
        self.tagCounter += 1
        return True

    def processDepthwiseSupervisorOperation(
            self,
            operation: DepthwiseSupervisorOperation):
        if (operation.childrenCompleted + operation.childrenInFlight != operation.nextChannel):
            raise Exception("ERROR! Depthwise supervisor child counters are inconsistent")

        allChildrenIssued = operation.nextChannel == channelsOut[operation.layer]
        if (allChildrenIssued and operation.childrenInFlight == 0):
            if (operation.childrenCompleted != channelsOut[operation.layer]):
                raise Exception("ERROR! Depthwise supervisor completed without all channels")
            pixelMap.setQuarterFinished(operation.layer, operation.quarter)
            return True

        while (operation.nextChannel < channelsOut[operation.layer]):
            if (not self.startDepthwiseChildOperation(operation, operation.nextChannel)):
                break
            operation.nextChannel += 1
            operation.childrenInFlight += 1

        return False

    def processDepthwiseConvOperation(self, operation: DepthwiseConvOperation):
        DSPunits = self.getDSPsWithOpTag(operation.tag)
        DSPGroups = [
            DSPunits[index:index + DSPS_PER_ACCUMULATOR]
            for index in range(0, len(DSPunits), DSPS_PER_ACCUMULATOR)
        ]
        if (
                len(DSPGroups) != operation.numDSPGroups
                or any(len(DSPGroup) != DSPS_PER_ACCUMULATOR for DSPGroup in DSPGroups)):
            raise Exception("ERROR! Depthwise operation has an invalid DSP allocation")

        quarterPixels = self.getDepthwiseQuarterPixels(operation.layer, operation.quarter)
        batchSize = min(
            operation.numDSPGroups,
            len(quarterPixels) - operation.batchStartPixel)
        if (batchSize <= 0):
            raise Exception("ERROR! Depthwise operation attempted to process an empty pixel batch")
        batchPixels = quarterPixels[
            operation.batchStartPixel:operation.batchStartPixel + batchSize
        ]

        if (operation.step == 0):
            if (not self.activationAskForKernel(
                    operation,
                    DSPGroups,
                    batchPixels)):
                return False

            for groupIndex, (outputX, outputY) in enumerate(batchPixels):
                kernelPartsByDSP = self.getDepthwiseDSPKernelParts(
                    operation.layer,
                    outputX,
                    outputY)
                for DSPoffset, kernelParts in enumerate(kernelPartsByDSP):
                    if (len(kernelParts) == 0):
                        continue
                    DSPunit = DSPGroups[groupIndex][DSPoffset]
                    if (not self.depthwiseKernelMatches(
                            DSPunit.preloadedKernel,
                            operation.layer,
                            operation.channel,
                            kernelParts)):
                        raise Exception("ERROR! Depthwise DSP is missing its kernel chunk")
                    if (not self.depthwiseActivationMatches(
                            DSPunit.preloadedActivationChunk,
                            operation.layer,
                            outputX,
                            outputY,
                            operation.channel,
                            kernelParts)):
                        raise Exception("ERROR! Depthwise DSP is missing its activation chunk")
                    DSPunit.loadKernel()
                    DSPunit.loadActivation()
                    DSPunit.startDotProduct()

            operation.step = 1
            return False

        if (operation.step == 1):
            if (any(DSPunit.isActive() for DSPGroup in DSPGroups[:batchSize] for DSPunit in DSPGroup)):
                return False
            operation.step = 2
            return False

        if (operation.step == 2):
            while (operation.nextOutputToStore < batchSize):
                groupIndex = operation.nextOutputToStore
                accumulator = DSPGroups[groupIndex][0].accumulator
                if (not accumulator.hasOutputReady()):
                    return False
                activation = accumulator.getOutput()
                if (
                        not isinstance(activation, Activation)
                        or activation.type != "depthwise"
                        or activation.layer != operation.layer
                        or activation.channel != operation.channel
                        or (activation.x, activation.y) != batchPixels[groupIndex]):
                    raise Exception("ERROR! Depthwise accumulator produced the wrong activation")
                if (not memory.loadActivationToSRAM(activation)):
                    return False
                accumulator.clearOutput()
                operation.nextOutputToStore += 1

            operation.batchStartPixel += batchSize
            if (operation.batchStartPixel == len(quarterPixels)):
                return True

            operation.step = 0
            operation.nextDSPToLoad = 0
            operation.nextOutputToStore = 0
            operation.kernelDistributedForBatch = False
            return False

        raise Exception("ERROR! Depthwise operation has an unrecognized step")

    def startDepthwiseConv(self, layer, quarter):
        if (layer <= 0 or layer >= layers or layerList[layer] != "depthwise"):
            raise Exception("ERROR! Invalid depthwise convolution layer")
        if (channelsIn[layer] != channelsOut[layer]):
            raise Exception("ERROR! Depthwise convolution input/output channels must match")
        if (channelsOut[layer - 1] != channelsIn[layer]):
            raise Exception("ERROR! Depthwise convolution input channels do not match its source layer")
        self.getDepthwiseQuarterPixels(layer, quarter)
        if (not self.isDepthwiseInputRegionFinished(layer, quarter)):
            return False
        if (pixelMap.isQuarterFinished(layer, quarter) or self.dwConvEnq(layer, quarter)):
            return False

        self.fifoPush(DepthwiseSupervisorOperation(
            layer=layer,
            tag=self.tagCounter,
            quarter=quarter))
        self.tagCounter += 1
        return True

    def startFullyConnected(self):
        if (self.fullyConnectedComplete or self.isFullyConnectedCalculating()):
            return False

        self.fifoPush(FullyConnectedOperation(tag=self.tagCounter))
        self.tagCounter += 1
        return True

    def startSEVectorCalc(self, layer):
        if (not pixelMap.isLayerFinished(layer)):
            # May want to remove this later to see how calculating SE vector with uncomplete averages
            # affects performance
            raise Exception("Trying to calculate SE Vector of unfinished layer")

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
            if isinstance(operation, CameraDepthwiseConvOperation):
                kernelRadius = DEPTHWISE_KERNEL_SIZE // 2
                cameraActivationList = []
                for height in range(DEPTHWISE_KERNEL_SIZE):
                    for width in range(DEPTHWISE_KERNEL_SIZE):
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
                    self.fifoRemove(fifoIndex)
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

                        DSPunit.preloadedKernel = replace(
                            DSPunit.preloadedKernel,
                            kernelParts=activationChunk.kernelParts,
                            valuesStored=activationChunk.valuesStored)
                        
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
                        # The squeeze sums and sample counts are a shared temporary
                        # buffer. Once this layer's complete SE vector is stored,
                        # reset them before a later SE layer begins accumulating.
                        memory.clearAccumulator()
                        if (operation.exciteTag is not None):
                            for DSPunit in self.getDSPsWithOpTag(operation.exciteTag):
                                DSPunit.setOperationTag(None)
                        self.fifoRemove(fifoIndex)
                        return

            if (isinstance(operation, FullyConnectedOperation)):
                fullyConnectedOperationComplete = self.processFullyConnectedOperation(operation)
                if (fullyConnectedOperationComplete):
                    self.fullyConnectedComplete = True
                    self.fifoRemove(fifoIndex)
                    return

            if (isinstance(operation, PointwiseSupervisorOperation)):
                pointwiseSupervisorComplete = self.processPointwiseSupervisorOperation(operation)
                if (pointwiseSupervisorComplete):
                    self.fifoRemove(fifoIndex)
                    return

            if (isinstance(operation, PointwiseConvOperation)):
                pointwiseOperationComplete = self.processPointwiseConvOperation(operation)
                if (pointwiseOperationComplete):
                    self.notifyPointwiseChildComplete(operation)
                    for DSPunit in self.getDSPsWithOpTag(operation.tag):
                        DSPunit.setOperationTag(None)
                    self.fifoRemove(fifoIndex)
                    return

            if (isinstance(operation, DepthwiseSupervisorOperation)):
                depthwiseSupervisorComplete = self.processDepthwiseSupervisorOperation(operation)
                if (depthwiseSupervisorComplete):
                    self.fifoRemove(fifoIndex)
                    return

            if (isinstance(operation, DepthwiseConvOperation)):
                depthwiseOperationComplete = self.processDepthwiseConvOperation(operation)
                if (depthwiseOperationComplete):
                    self.notifyDepthwiseChildComplete(operation)
                    for DSPunit in self.getDSPsWithOpTag(operation.tag):
                        DSPunit.setOperationTag(None)
                    self.fifoRemove(fifoIndex)
                    return

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
MAX_SQUEEZE_EXCITE_CHANNELS = 244
MAX_CHANNEL_AVERAGE_CHANNELS = max(
    MAX_SQUEEZE_EXCITE_CHANNELS,
    int(np.max(channelsOut)))
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
        # Strided pointwise reads can span multiple addresses in one LRAM.
        # Preserve completed reads across cycles instead of exceeding its port limit.
        self.pendingPointwiseKernelReads = {}

        self.squeezeExciteAccumulator = np.zeros(MAX_CHANNEL_AVERAGE_CHANNELS)
        self.squeezeExciteAccumulatorCount = np.zeros(MAX_CHANNEL_AVERAGE_CHANNELS, dtype=int)
        self.SEHiddenLayerVector = SEHiddenLayerVector(length=0)
        
        self.operationFIFO = deque()
        self.memTagPushCounter = 0
        self.memTagPopCounter = 0

    def addActivationToSqueezeExciteAccumulator(self, activation: Activation):
        nextLayerIsGAP = (
            activation.layer + 1 < layers
            and layerList[activation.layer + 1] == "gap")
        if (squeezeExcite[activation.layer] != 1 and not nextLayerIsGAP):
            return
        if (activation.channel < 0 or activation.channel >= MAX_CHANNEL_AVERAGE_CHANNELS):
            raise Exception("ERROR! Activation channel is outside the channel-average accumulator range")

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
        if (len(SEVectorIndexList) <= 0 or len(SEVectorIndexList) > DSP_CHUNK_SIZE):
            raise Exception(f"ERROR! SE vector chunk must be made from 1 to {DSP_CHUNK_SIZE} SE vector indexes")
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

    def getSEWeightChunksForDSP(
            self,
            layer,
            isExpansion,
            dspIndex,
            weightsPerDSP,
            weightsLoaded,
            weightsToLoad,
            outputChannelStart=0):
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
        if (isExpansion):
            row += outputChannelStart
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
            weights[index:index + DSP_CHUNK_SIZE]
            for index in range(0, len(weights), DSP_CHUNK_SIZE)
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
            SEVectorIndexes[index:index + DSP_CHUNK_SIZE]
            for index in range(0, len(SEVectorIndexes), DSP_CHUNK_SIZE)
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
            SEhiddenLayerElems[index:index + DSP_CHUNK_SIZE]
            for index in range(0, len(SEhiddenLayerElems), DSP_CHUNK_SIZE)
        ]

        return [
            SEVectorChunk(
                type="input",
                layer=layer,
                channel=SEhiddenLayerElemChunk[0].elemNumber,
                valuesStored=len(SEhiddenLayerElemChunk))
            for SEhiddenLayerElemChunk in SEhiddenLayerElemChunks
        ]

    def getFullyConnectedWeightChunksForDSP(
            self,
            stage,
            dspIndex,
            weightsPerDSP,
            weightsLoaded,
            weightsToLoad,
            outputChannelStart):
        if (stage < 0 or stage > 1):
            raise Exception("ERROR! Fully connected stage must be 0 or 1")
        if (dspIndex < 0 or dspIndex >= NUM_DSPS):
            raise Exception("ERROR! Fully connected DSP index is outside the DSP range")
        if (weightsPerDSP <= 0 or weightsLoaded < 0 or weightsToLoad <= 0):
            raise Exception("ERROR! Fully connected weight request is invalid")

        layer = FULLY_CONNECTED_LAYER_START + stage
        inputVectorLength = FULLY_CONNECTED_INPUT if stage == 0 else FULLY_CONNECTED_HIDDEN
        outputVectorLength = FULLY_CONNECTED_HIDDEN if stage == 0 else FULLY_CONNECTED_OUTPUT
        DSPsPerRow = math.ceil(inputVectorLength / weightsPerDSP)
        row = outputChannelStart + (dspIndex // DSPsPerRow)
        dspIndexInRow = dspIndex % DSPsPerRow
        columnStart = dspIndexInRow * weightsPerDSP + weightsLoaded
        if (row < 0 or row >= outputVectorLength):
            raise Exception("ERROR! Fully connected weight row is outside the output vector")
        if (columnStart < 0 or columnStart + weightsToLoad > inputVectorLength):
            raise Exception("ERROR! Fully connected weight request crosses a matrix row")

        trueOffset = row * inputVectorLength + columnStart
        layerLRAMStart, layerAddrStart, layerPackedStart = self.startOfLayer[layer]
        packedOffset = int(layerPackedStart) + trueOffset
        LRAMWordOffset, packedIndex = divmod(packedOffset, WEIGHTS_PER_LRAM_ADDRESS)
        logicalLRAM = int(layerLRAMStart) + LRAMWordOffset
        LRAMNum = logicalLRAM % NUM_LRAMS
        LRAMAddr = int(layerAddrStart) + (logicalLRAM // NUM_LRAMS)

        weights = []
        if (not self.readSequentially(LRAMNum, LRAMAddr, packedIndex, weightsToLoad, weights)):
            return False
        for offset, weight in enumerate(weights):
            if (
                    not isinstance(weight, FullyConnectedWeight)
                    or weight.layer != layer
                    or weight.row != row
                    or weight.column != columnStart + offset):
                raise Exception("ERROR! LRAM returned the wrong fully connected weight")

        return [
            SEKernel(weights=weights[index:index + DSP_CHUNK_SIZE])
            for index in range(0, len(weights), DSP_CHUNK_SIZE)
        ]

    def getFullyConnectedVectorLocation(self, vectorIndex, isFinalOutput=False):
        vectorLength = FULLY_CONNECTED_OUTPUT if isFinalOutput else FULLY_CONNECTED_HIDDEN
        if (vectorIndex < 0 or vectorIndex >= vectorLength):
            raise Exception("ERROR! Fully connected vector index is outside the SRAM scratch range")
        scratchOffset = FULLY_CONNECTED_HIDDEN if isFinalOutput else 0
        if (scratchOffset + vectorIndex >= MAX_SQUEEZE_EXCITE_CHANNELS):
            raise Exception("ERROR! Fully connected vectors exceed the reusable SE SRAM region")
        trueIndex = SEWEIGHTTOTALBYTES + scratchOffset + vectorIndex
        packedIndex = trueIndex % SE_WEIGHTS_PER_SRAM_ADDRESS
        SRAMJumps = trueIndex // SE_WEIGHTS_PER_SRAM_ADDRESS
        return (
            SRAMJumps % NUM_SRAMS,
            SRAMJumps // NUM_SRAMS,
            packedIndex)

    def loadFullyConnectedVectorPart(self, vectorPart: SEVectorIndex):
        if (not isinstance(vectorPart, SEVectorIndex)):
            raise Exception("ERROR! Fully connected output must be a vector index")
        if (vectorPart.layer not in (FULLY_CONNECTED_LAYER_START, FULLY_CONNECTED_LAYER_START + 1)):
            raise Exception("ERROR! Fully connected vector part has the wrong layer")
        expectedType = "input" if vectorPart.layer == FULLY_CONNECTED_LAYER_START else "output"
        expectedLength = (
            FULLY_CONNECTED_HIDDEN
            if vectorPart.layer == FULLY_CONNECTED_LAYER_START
            else FULLY_CONNECTED_OUTPUT)
        if (vectorPart.type != expectedType or vectorPart.channel < 0 or vectorPart.channel >= expectedLength):
            raise Exception("ERROR! Fully connected vector part has invalid metadata")

        SRAMNum, SRAMAddr, packedIndex = self.getFullyConnectedVectorLocation(
            vectorPart.channel,
            isFinalOutput=(vectorPart.layer == FULLY_CONNECTED_LAYER_START + 1))
        SRAMobj = self.allSRAM[SRAMNum]
        if (SRAMobj.isActiveRead() or SRAMobj.isActiveWrite()):
            return False
        packedParts = list(SRAMobj.readSRAM(SRAMAddr))
        packedParts[packedIndex] = vectorPart
        SRAMobj.writeSRAM(SRAMAddr, packedParts)
        return True

    def readFullyConnectedHiddenVectorParts(self, firstIndex, valuesToRead):
        if (firstIndex < 0 or valuesToRead <= 0 or firstIndex + valuesToRead > FULLY_CONNECTED_HIDDEN):
            raise Exception("ERROR! Fully connected hidden-vector read is outside the SRAM scratch range")

        locations = [
            self.getFullyConnectedVectorLocation(index)
            for index in range(firstIndex, firstIndex + valuesToRead)
        ]
        SRAMWords = {(SRAMNum, SRAMAddr) for SRAMNum, SRAMAddr, _ in locations}
        if (any(self.allSRAM[SRAMNum].isActiveRead() for SRAMNum, _ in SRAMWords)):
            return False

        packedWords = {
            (SRAMNum, SRAMAddr): self.allSRAM[SRAMNum].readSRAM(SRAMAddr)
            for SRAMNum, SRAMAddr in SRAMWords
        }
        vectorParts = []
        for expectedIndex, (SRAMNum, SRAMAddr, packedIndex) in enumerate(locations, start=firstIndex):
            vectorPart = packedWords[(SRAMNum, SRAMAddr)][packedIndex]
            if (
                    not isinstance(vectorPart, SEVectorIndex)
                    or vectorPart.type != "input"
                    or vectorPart.layer != FULLY_CONNECTED_LAYER_START
                    or vectorPart.channel != expectedIndex):
                raise Exception("ERROR! SRAM returned the wrong fully connected hidden-vector part")
            vectorParts.append(vectorPart)
        return vectorParts

    def getFullyConnectedInputChunksForDSP(
            self,
            stage,
            dspIndex,
            valuesPerDSP,
            valuesLoaded,
            valuesToLoad):
        if (stage < 0 or stage > 1):
            raise Exception("ERROR! Fully connected stage must be 0 or 1")
        if (dspIndex < 0 or dspIndex >= NUM_DSPS):
            raise Exception("ERROR! Fully connected DSP index is outside the DSP range")
        if (valuesPerDSP <= 0 or valuesLoaded < 0 or valuesToLoad <= 0):
            raise Exception("ERROR! Fully connected input request is invalid")

        inputVectorLength = FULLY_CONNECTED_INPUT if stage == 0 else FULLY_CONNECTED_HIDDEN
        DSPsPerRow = math.ceil(inputVectorLength / valuesPerDSP)
        dspIndexInRow = dspIndex % DSPsPerRow
        firstInputIndex = dspIndexInRow * valuesPerDSP + valuesLoaded
        if (firstInputIndex + valuesToLoad > inputVectorLength):
            raise Exception("ERROR! Fully connected input request crosses an input vector")

        if (stage == 0):
            vectorParts = [
                self.readAveragedChannel(GAP_INPUT_LAYER, channel)
                for channel in range(firstInputIndex, firstInputIndex + valuesToLoad)
            ]
        else:
            vectorParts = self.readFullyConnectedHiddenVectorParts(firstInputIndex, valuesToLoad)
            if (vectorParts == False):
                return False

        return [
            self.SEVectorIndexToSEVectorChunk(vectorParts[index:index + DSP_CHUNK_SIZE])
            for index in range(0, len(vectorParts), DSP_CHUNK_SIZE)
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
        if (not weightList):
            raise Exception("ERROR! Cannot make kernel from an empty weight list")
        if (any(weight is None for weight in weightList)):
            raise Exception("ERROR! Cannot make kernel from empty weight")

        firstWeight = weightList[0]

        if (isinstance(firstWeight, DepthwiseWeight)):
            if (len(weightList) != DEPTHWISE_KERNEL_PARTS):
                raise Exception("ERROR! Depthwise kernel must be made from exactly 9 weights")
            if (firstWeight.part != 0):
                raise Exception("ERROR! Depthwise kernel must start at part 0")

            for idx, weight in enumerate(weightList):
                if (not isinstance(weight, DepthwiseWeight)):
                    raise Exception("ERROR! Cannot make kernel from mixed weight types")
                if (weight.layer != firstWeight.layer or weight.channel != firstWeight.channel):
                    raise Exception("ERROR! Depthwise weights are not in the same kernel")
                if (weight.part != idx):
                    raise Exception("ERROR! Depthwise weights are not sequential")

            return DepthwiseKernel(
                layer=firstWeight.layer,
                channel=firstWeight.channel,
                kernelParts=tuple(range(DEPTHWISE_KERNEL_PARTS)),
                valuesStored=DEPTHWISE_KERNEL_PARTS)

        if (isinstance(firstWeight, PointwiseWeight)):
            if (len(weightList) > DSP_CHUNK_SIZE):
                raise Exception(f"ERROR! Pointwise kernel cannot contain more than {DSP_CHUNK_SIZE} weights")
            for weight in weightList:
                if (not isinstance(weight, PointwiseWeight)):
                    raise Exception("ERROR! Cannot make kernel from mixed weight types")
                if (weight.layer != firstWeight.layer or weight.y != firstWeight.y):
                    raise Exception("ERROR! Pointwise weights are not in the same kernel")
            inputChannels = tuple(weight.channel for weight in weightList)
            if (inputChannels != tuple(sorted(set(inputChannels)))):
                raise Exception("ERROR! Pointwise weight channels must be distinct and increasing")

            return PointwiseKernel(
                layer=firstWeight.layer,
                channel=firstWeight.y,
                inputChannels=inputChannels)

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
            kernelRadius = DEPTHWISE_KERNEL_SIZE // 2
            layerSpatialSize = spatialInput[firstActivation.layer]

            for activation in activationList:
                if (activation.layer != firstActivation.layer or activation.channel != firstActivation.channel):
                    raise Exception("ERROR! Depthwise activations are not in the same chunk")
                if (activation.x < 0 or activation.x >= layerSpatialSize):
                    raise Exception(f"ERROR! Depthwise activation x coordinate is outside the layer bounds.\n X val = {activation.x}")
                if (activation.y < 0 or activation.y >= layerSpatialSize):
                    raise Exception("ERROR! Depthwise activation y coordinate is outside the layer bounds")

            activationCoordinates = [(activation.x, activation.y) for activation in activationList]
            possibleTopLeftCoordinates = []
            for topLeftY in range(firstActivation.y - DEPTHWISE_KERNEL_SIZE + 1, firstActivation.y + 1):
                for topLeftX in range(firstActivation.x - DEPTHWISE_KERNEL_SIZE + 1, firstActivation.x + 1):
                    expectedCoordinates = []
                    for kernelY in range(DEPTHWISE_KERNEL_SIZE):
                        for kernelX in range(DEPTHWISE_KERNEL_SIZE):
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
            kernelParts = tuple(
                kernelY * DEPTHWISE_KERNEL_SIZE + kernelX
                for kernelY in range(DEPTHWISE_KERNEL_SIZE)
                for kernelX in range(DEPTHWISE_KERNEL_SIZE)
                if (
                    0 <= topLeftX + kernelX < layerSpatialSize
                    and 0 <= topLeftY + kernelY < layerSpatialSize)
            )

            return ActivationChunk(
                type=firstActivation.type,
                layer=firstActivation.layer,
                x=topLeftX + kernelRadius,
                y=topLeftY + kernelRadius,
                channel=firstActivation.channel,
                kernelParts=kernelParts,
                valuesStored=len(kernelParts))

        if (firstActivation.type == "pointwise"):
            if (not 0 < len(activationList) <= DSP_CHUNK_SIZE):
                raise Exception("ERROR! Pointwise activation chunk has incorrect number of activations")

            inputChannels = tuple(activation.channel for activation in activationList)
            lowestChannel = inputChannels[0]
            if (inputChannels != tuple(sorted(set(inputChannels)))):
                raise Exception("ERROR! Pointwise activation channels must be distinct and increasing")

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
                valuesStored=len(inputChannels),
                inputChannels=inputChannels)

        raise Exception("ERROR! Unsupported activation type for activation chunk conversion")

    def askForDepthwiseActivationChunk(
            self,
            layer,
            outputX,
            outputY,
            channel,
            kernelParts,
            id):
        if (layer <= 0 or layer >= layers or layerList[layer] != "depthwise"):
            raise Exception("ERROR! Invalid depthwise activation chunk layer")
        if (channel < 0 or channel >= channelsIn[layer]):
            raise Exception("ERROR! Invalid depthwise activation channel")
        if (len(kernelParts) == 0 or len(kernelParts) > 3):
            raise Exception("ERROR! Depthwise DSP chunk must contain between one and three values")

        sourceLayer = layer - 1
        inputCenterX = outputX * stride[layer]
        inputCenterY = outputY * stride[layer]
        kernelRadius = DEPTHWISE_KERNEL_SIZE // 2
        requestedActivations = []
        for kernelPart in kernelParts:
            kernelX = kernelPart % DEPTHWISE_KERNEL_SIZE
            kernelY = kernelPart // DEPTHWISE_KERNEL_SIZE
            inputX = inputCenterX + kernelX - kernelRadius
            inputY = inputCenterY + kernelY - kernelRadius
            if (
                    inputX < 0 or inputX >= spatialOutput[sourceLayer]
                    or inputY < 0 or inputY >= spatialOutput[sourceLayer]):
                raise Exception("ERROR! Depthwise chunk requested a padded activation")
            requestedActivations.append(Activation(
                type=layerList[sourceLayer],
                layer=sourceLayer,
                x=inputX,
                y=inputY,
                channel=channel))

        activationLocations = [
            self.getActivationSRAMLocation(activation)
            for activation in requestedActivations
        ]
        SRAMWords = []
        for SRAMNum, SRAMAddr, _ in activationLocations:
            SRAMWord = (SRAMNum, SRAMAddr)
            if (SRAMWord not in SRAMWords):
                SRAMWords.append(SRAMWord)

        SRAMNumbers = [SRAMNum for SRAMNum, _ in SRAMWords]
        if (len(SRAMNumbers) != len(set(SRAMNumbers))):
            raise Exception("ERROR! Depthwise activation chunk requires multiple reads from one SRAM")
        for SRAMNum, _ in SRAMWords:
            SRAMobj = self.allSRAM[SRAMNum]
            if (SRAMobj.isActiveRead() or SRAMobj.isActiveWrite()):
                return False

        packedWords = {
            (SRAMNum, SRAMAddr): self.allSRAM[SRAMNum].readSRAM(SRAMAddr)
            for SRAMNum, SRAMAddr in SRAMWords
        }
        for requestedActivation, (SRAMNum, SRAMAddr, packedIndex) in zip(
                requestedActivations,
                activationLocations):
            storedActivation = packedWords[(SRAMNum, SRAMAddr)][packedIndex]
            if (not isinstance(storedActivation, Activation)):
                raise Exception("ERROR! Depthwise activation was not present in SRAM")
            if (
                    storedActivation.type != requestedActivation.type
                    or storedActivation.layer != requestedActivation.layer
                    or storedActivation.x != requestedActivation.x
                    or storedActivation.y != requestedActivation.y
                    or storedActivation.channel != requestedActivation.channel):
                raise Exception("ERROR! SRAM returned the wrong depthwise activation")

        DSPunit = DSPMngr.getDSPByID(id)
        DSPunit.preloadActivation(ActivationChunk(
            type="depthwise",
            layer=layer,
            x=inputCenterX,
            y=inputCenterY,
            channel=channel,
            kernelParts=tuple(kernelParts),
            valuesStored=len(kernelParts)))
        return True

    def askForPointwiseActivationChunk(self, layer, x, y, inputChannels, id):
        if (layer <= 0 or layer >= layers):
            raise Exception("ERROR! Pointwise activation layer is outside the readable layer range")
        if (layerList[layer] != "pointwise"):
            raise Exception("ERROR! Pointwise activation chunk requested for a non-pointwise layer")
        if (channelsOut[layer - 1] != channelsIn[layer]):
            raise Exception("ERROR! Pointwise input channel count does not match the preceding layer")
        if (not 0 < len(inputChannels) <= DSP_CHUNK_SIZE
                or inputChannels != tuple(sorted(set(inputChannels)))
                or inputChannels[0] < 0
                or inputChannels[-1] >= channelsIn[layer]):
            raise Exception("ERROR! Pointwise activation chunk has invalid input channels")

        sourceLayer = layer - 1
        requestedActivations = [
            Activation(
                type=layerList[sourceLayer],
                layer=sourceLayer,
                x=x,
                y=y,
                channel=channel)
            for channel in inputChannels
        ]
        activationLocations = [
            self.getActivationSRAMLocation(activation)
            for activation in requestedActivations
        ]
        SRAMWords = []
        for SRAMNum, SRAMAddr, _ in activationLocations:
            SRAMWord = (SRAMNum, SRAMAddr)
            if (SRAMWord not in SRAMWords):
                SRAMWords.append(SRAMWord)

        SRAMNumbers = [SRAMNum for SRAMNum, _ in SRAMWords]
        if (len(SRAMNumbers) != len(set(SRAMNumbers))):
            raise Exception("ERROR! Pointwise activation chunk requires multiple reads from one SRAM")
        for SRAMNum, _ in SRAMWords:
            SRAMobj = self.allSRAM[SRAMNum]
            if (SRAMobj.isActiveRead() or SRAMobj.isActiveWrite()):
                return False

        packedWords = {}
        for SRAMNum, SRAMAddr in SRAMWords:
            packedWords[(SRAMNum, SRAMAddr)] = self.allSRAM[SRAMNum].readSRAM(SRAMAddr)

        for requestedActivation, (SRAMNum, SRAMAddr, packedIndex) in zip(
                requestedActivations,
                activationLocations):
            storedActivation = packedWords[(SRAMNum, SRAMAddr)][packedIndex]
            if (not isinstance(storedActivation, Activation)):
                raise Exception("ERROR! Pointwise activation was not present in SRAM")
            if (
                    storedActivation.layer != requestedActivation.layer
                    or storedActivation.x != requestedActivation.x
                    or storedActivation.y != requestedActivation.y
                    or storedActivation.channel != requestedActivation.channel):
                raise Exception("ERROR! SRAM returned the wrong pointwise activation")

        DSPunit = DSPMngr.getDSPByID(id)
        DSPunit.preloadActivation(ActivationChunk(
            type="pointwise",
            layer=layer,
            x=x,
            y=y,
            channel=inputChannels[0],
            valuesStored=len(inputChannels),
            inputChannels=inputChannels))
        return True

    def askForKernel(self, kernel: Kernel, id):
        # Locate spatial depthwise weights or the explicitly selected pointwise
        # input channels within the layer's stored weights.
        
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
            # Fetch the full spatial kernel; the depthwise scheduler splits it
            # into the smaller operands assigned to individual DSPs.
            
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
            # Pointwise weights are stored output-channel major. Each output
            # channel contains sequential input-channel weights. Gather only
            # this DSP's channels, retaining progress when an LRAM is busy.
            if (not 0 < len(kernel.inputChannels) <= DSP_CHUNK_SIZE
                    or kernel.inputChannels != tuple(sorted(set(kernel.inputChannels)))
                    or kernel.inputChannels[0] < 0
                    or kernel.inputChannels[-1] >= channelsIn[kernel.layer]
                    or kernel.channel < 0 or kernel.channel >= channelsOut[kernel.layer]):
                raise Exception("ERROR! Pointwise kernel has invalid input or output channels")

            pendingRead = self.pendingPointwiseKernelReads.get(id)
            if (pendingRead is None or pendingRead[0] != kernel):
                pendingRead = (replace(kernel), [])
                self.pendingPointwiseKernelReads[id] = pendingRead
            weights = pendingRead[1]
            layerLRAMStart, layerAddrStart, layerPackedStart = self.startOfLayer[kernel.layer]

            for inputChannel in kernel.inputChannels[len(weights):]:
                packedOffset = (
                    int(layerPackedStart)
                    + kernel.channel * int(channelsIn[kernel.layer])
                    + inputChannel)
                wordOffset, packedIndex = divmod(packedOffset, WEIGHTS_PER_LRAM_ADDRESS)
                logicalLRAM = int(layerLRAMStart) + wordOffset
                LRAMNum = logicalLRAM % NUM_LRAMS
                LRAMAddr = int(layerAddrStart) + logicalLRAM // NUM_LRAMS
                weightRead = []
                if (not self.readSequentially(LRAMNum, LRAMAddr, packedIndex, 1, weightRead)):
                    return False
                weights.extend(weightRead)

            returnedKernel = self.weightsToKernel(weights)
            if (returnedKernel != kernel):
                raise Exception(
                    f"ERROR! Memory returned the wrong kernel. Desired kernel: {kernel}, "
                    f"returned kernel: {returnedKernel}")

            dsp = DSPMngr.getDSPByID(id)
            dsp.preloadKernel(returnedKernel)
            del self.pendingPointwiseKernelReads[id]
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

# Store each pointwise matrix in output-channel-major order. Chunk boundaries
# belong to the DSP scheduler; they do not change the weight layout in memory.
# Depthwise kernels contain all spatial weights for each channel in turn.
for layer in range(layers - NUM_MISC_LAYERS):
    memory.markLayerStart(layer)
    if (layerList[layer] == "depthwise"):
        layerWeights = (
            DepthwiseWeight(layer=layer, channel=channel, part=part)
            for channel in range(channelsOut[layer])
            for part in range(DEPTHWISE_KERNEL_PARTS)
        )
    elif (layerList[layer] == "pointwise"):
        layerWeights = (
            PointwiseWeight(layer=layer, y=outputChannel, channel=inputChannel)
            for outputChannel in range(channelsOut[layer])
            for inputChannel in range(channelsIn[layer])
        )
    else:
        raise Exception("ERROR! Unsupported convolution weight type")

    weightsLoadedThisCycle = 0
    for weight in layerWeights:
        memory.loadWeightsInStart(weight)
        weightsLoadedThisCycle += 1
        if (weightsLoadedThisCycle == MAX_SIMULTANEOUS_LRAM_WRITES * WEIGHTS_PER_LRAM_ADDRESS):
            clkCycles += 1
            weightsLoadedThisCycle = 0
    if (weightsLoadedThisCycle > 0):
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

SIM_LIMIT = 2000000

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

            kernelRadius = DEPTHWISE_KERNEL_SIZE // 2
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
        print(f"cycles: {clkCycles}, performing SE, lastCompletedLayer: {lastCompletedLayer}\n")
        if (not DSPMngr.isSEVectorCalculating()):
            DSPMngr.startSEVectorCalc(lastCompletedLayer)
        elif (pixelMap.isSEVectorCalculated(lastCompletedLayer)):
            pass
            # I think the only way to really implement attention efficiently is with bit-shifts
            # There is no activation after attention is applied
    elif (not squeezeExcite[lastCompletedLayer] or (squeezeExcite[lastCompletedLayer] and pixelMap.isSEVectorCalculated(lastCompletedLayer))):
        print(f"cycles: {clkCycles}, performing conv, lastCompletedLayer: {lastCompletedLayer}\n")
        for layer in range(lastCompletedLayer + 1, lastCompletedLayer + 3):
            for quarter in range(1, SEGMENTS_PER_LAYER + 1):
                if (layerList[layer] == "pointwise"):
                    if (pixelMap.isQuarterFinished(layer - 1, quarter)
                        and not pixelMap.isQuarterFinished(layer, quarter)
                        and not DSPMngr.pwConvEnq(layer, quarter)):
                            DSPMngr.startPointwiseConv(
                                layer,
                                quarter,
                                completedQuarters=pixelMap.getCompletedQuarters(layer))
                if (layer > 0 and layerList[layer] == "depthwise"):
                    if (pixelMap.isQuarterFinished(layer - 1, quarter)
                        and not pixelMap.isQuarterFinished(layer, quarter)
                        and not DSPMngr.dwConvEnq(layer, quarter)):
                            DSPMngr.startDepthwiseConv(layer, quarter)
                if (lastCompletedLayer == 34):
                    DSPMngr.startFullyConnected()
    # Fully Connected Neural Net Logic
    
    DSPMngr.update()
    if (DSPMngr.fullyConnectedComplete):
        inferenceCompleted = True
    clkCycles += 1

exportToJSON(memory, "done with inference")
