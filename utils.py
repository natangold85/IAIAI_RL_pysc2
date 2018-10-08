import random
import math
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib.units import Terran

import numpy as np

#base class for all shared data classes
class EmptySharedData:
    def __init__(self):
        return

class EmptyLock:
    def acquire(self):
        return
    def release(self):
        return

class BaseAgent(base_agent.BaseAgent):
    def __init__(self, stateSize = None):
        super(BaseAgent, self).__init__()
        
        if stateSize != None:
            self.terminalState = np.zeros(stateSize, int)
        
        self.decisionMaker = None
        self.history = None
        self.trainAgent = False
        self.subAgents = {}
    
    def GetDecisionMaker(self):
        return None
    
    def FindActingHeirarchi(self):
        return -1

    def CreateState(self, obs):
        pass

    def MonitorObservation(self, obs):
        pass

    def FirstStep(self, obs = None):
        self.isActionCommitted = False

        self.current_action = None
        self.lastActionCommitted = None

        if self.decisionMaker != None:
            self.minReward = self.decisionMaker.GetMinReward()
            self.maxReward = self.decisionMaker.GetMaxReward()
        else:
            self.minReward = 0
            self.maxReward = 0


    def EndRun(self, reward, score, stepNum):
        pass
    
    def Learn(self, reward = 0, terminal = False):
        pass

    def Action2SC2Action(self, obs, a, moveNum):
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def IsDoNothingAction(self, a):
        return True

    def Action2Str(self, a):
        return "None"

    def StateIdx2Str(self, idx):
        return "None"
    
    def SubAgentActionChosen(self, action):
        self.isActionCommitted = True
        self.lastActionCommitted = action

    def GetStateVal(self, idx):
        return None

    def RemoveNonTerminalHistory(self):
        if self.history != None:
            self.history.RemoveNonTerminalHistory()    

    def NormalizeReward(self, reward):
        return 2 * (reward - self.minReward) / (self.maxReward - self.minReward) - 1
        
    def AddTerminalReward(self, reward):
        for sa in self.subAgents.values():
            sa.AddTerminalReward()

        if self.trainAgent:
            if reward < self.minReward:
                self.minReward = reward
                self.decisionMaker.SetMinReward(reward)
            elif reward > self.maxReward:
                self.maxReward = reward
                self.decisionMaker.SetMaxReward(reward)    

# params base
class ParamsBase:
    def __init__(self, stateSize, numActions, discountFactor = 0.95, maxReplaySize = 500000, minReplaySize = 1000):
        self.stateSize = stateSize
        self.numActions = numActions
        self.discountFactor = discountFactor
        self.maxReplaySize = maxReplaySize
        self.minReplaySize = minReplaySize

class SC2_Params:
    # minimap feature
    CAMERA = features.MINIMAP_FEATURES.camera.index
    HEIGHT_MINIMAP = features.MINIMAP_FEATURES.height_map.index
    VISIBILITY_MINIMAP = features.MINIMAP_FEATURES.visibility_map.index
    PLAYER_RELATIVE_MINIMAP = features.MINIMAP_FEATURES.player_relative.index
    SELECTED_IN_MINIMAP = features.MINIMAP_FEATURES.selected.index

    # screen feature
    HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
    PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
    
    #HIT_POINTS = features.SCREEN_FEATURES.hit_points.index
    UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
    PLAYER_ID = features.SCREEN_FEATURES.player_id.index
    SELECTED_IN_SCREEN = features.SCREEN_FEATURES.selected.index
    VISIBILITY = features.SCREEN_FEATURES.visibility_map.index 
    
    # visibility map details
    IN_SIGHT = 2
    SLIGHT_FOG = 1
    FOG = 0

    # player id type
    PLAYER_SELF = 1
    PLAYER_NEUTRAL = 3 
    PLAYER_HOSTILE = 4

    #player general info
    MINERALS = 1
    VESPENE = 2
    SUPPLY_USED = 3
    SUPPLY_CAP = 4
    ARMY_SUPPLY_OCCUPATION = 5
    WORKERS_SUPPLY_OCCUPATION = 6
    IDLE_WORKER_COUNT = 7

    # single, multi and building queue select table idx
    UNIT_TYPE_IDX = 0
    COMPLETION_RATIO_IDX = 6

    # params for queued argument
    NOT_QUEUED = [0]
    QUEUED = [1]

    # params for select 
    SELECT_SINGLE = [0]
    SELECT_ALL = [2]

    # params for control group
    CONTROL_GROUP_RECALL = [0]
    CONTROL_GROUP_SET = [1]
    CONTROL_GROUP_APPEND = [2]
    CONTROL_GROUP_SET_AND_STEAL = [3]
    CONTROL_GROUP_APPEND_AND_STEAL = [4]

    # control groups info
    LEADING_UNIT_CONTROL_GROUP = 0
    NUM_UNITS_CONTROL_GROUP = 1

    NEUTRAL_MINERAL_FIELD = [341, 483]
    VESPENE_GAS_FIELD = [342]

    Y_IDX = 0
    X_IDX = 1

    MINIMAP_SIZE = 64
    SCREEN_SIZE = 84
    MAX_MINIMAP_DIST = MINIMAP_SIZE * MINIMAP_SIZE + MINIMAP_SIZE * MINIMAP_SIZE


    TOPLEFT_BASE_LOCATION = [23,18]
    BOTTOMRIGHT_BASE_LOCATION = [45,39]

class SC2_Actions:
    # general actions
    NO_OP = actions.FUNCTIONS.no_op.id
    SELECT_POINT = actions.FUNCTIONS.select_point.id
    SELECT_RECTANGLE = actions.FUNCTIONS.select_rect.id
    STOP = actions.FUNCTIONS.Stop_quick.id

    MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
    HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
    SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id

    # build actions
    BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
    BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
    BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
    BUILD_OIL_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id

    # building additions
    BUILD_REACTOR_QUICK = actions.FUNCTIONS.Build_Reactor_Barracks_quick.id 
    BUILD_TECHLAB_QUICK = actions.FUNCTIONS.Build_TechLab_quick.id 

    BUILD_REACTOR = actions.FUNCTIONS.Build_Reactor_screen.id
    BUILD_TECHLAB = actions.FUNCTIONS.Build_TechLab_screen.id

    # train actions
    TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
    RALLY_SCV = actions.FUNCTIONS.Rally_Workers_screen.id
    TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
    TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

    TRAIN_HELLION = actions.FUNCTIONS.Train_Hellion_quick.id
    TRAIN_SIEGE_TANK = actions.FUNCTIONS.Train_SiegeTank_quick.id

    # queue
    BUILD_QUEUE = actions.FUNCTIONS.build_queue.id
    
    SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

    SELECT_ARMY = actions.FUNCTIONS.select_army.id
    MOVE_IN_SCREEN = actions.FUNCTIONS.Move_screen.id
    ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
    ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id

    STOP_SC2_ACTION = actions.FunctionCall(STOP, [SC2_Params.NOT_QUEUED])
    DO_NOTHING_SC2_ACTION = actions.FunctionCall(NO_OP, [])







class TerranUnit:
    BUILDINGS = [Terran.Armory, Terran.AutoTurret, Terran.Barracks, Terran.BarracksReactor, Terran.BarracksTechLab, Terran.Bunker, Terran.CommandCenter, Terran.Cyclone, 
                Terran.EngineeringBay, Terran.Factory, Terran.FactoryReactor, Terran.FactoryTechLab, Terran.FusionCore, Terran.GhostAcademy, Terran.MissileTurret, 
                Terran.OrbitalCommand, Terran.PlanetaryFortress, Terran.Reactor, Terran.Refinery, Terran.SensorTower, Terran.Starport, Terran.StarportReactor, 
                Terran.StarportTechLab, Terran.SupplyDepot, Terran.SupplyDepotLowered, Terran.TechLab]

          
    ARMY= [Terran.Banshee, Terran.Ghost, Terran.Battlecruiser, Terran.Hellion, Terran.Hellbat, Terran.Liberator, Terran.LiberatorAG, Terran.Marauder, 
            Terran.Marine, Terran.Medivac, Terran.Raven, Terran.Reaper, Terran.SCV, Terran.SiegeTank, Terran.SiegeTankSieged, Terran.Thor, Terran.ThorHighImpactMode, 
            Terran.VikingAssault, Terran.VikingFighter]

        
    FLYING_BUILDINGS = [Terran.BarracksFlying, Terran.CommandCenterFlying, Terran.FactoryFlying, Terran.OrbitalCommandFlying, Terran.StarportFlying]
    
    ALL_RESOURCES = SC2_Params.NEUTRAL_MINERAL_FIELD + SC2_Params.VESPENE_GAS_FIELD + [Terran.Refinery]

    BLOCKING_TYPE = {}
    BLOCKING_TYPE[0] = False
    for building in BUILDINGS:
        BLOCKING_TYPE[building] = True
    for resource in ALL_RESOURCES:
        BLOCKING_TYPE[resource] = True  

    for army in ARMY:
        BLOCKING_TYPE[army] = False
    for flying in FLYING_BUILDINGS:
        BLOCKING_TYPE[flying] = False     

    # building specific:
    class BuildingDetails:
        def __init__(self,name, screenPixels, screenPixels1Axis, char4Print, miniMapSize, sc2Action = None):
            self.name = name
            self.numScreenPixels = screenPixels
            self.screenPixels1Axis = screenPixels1Axis
            self.char4Print = char4Print   
            self.miniMapSize = miniMapSize    
            self.sc2Action = sc2Action
    
    # army specific:

    class UnitDetails:
        def __init__(self,name, screenPixels, foodCapacity, char4Print, sc2Action = None):
            self.name = name
            self.numScreenPixels = screenPixels
            self.foodCapacity = foodCapacity
            self.char4Print = char4Print
            self.sc2Action = sc2Action

            
    
    BUILDING_SPEC = {}
    # buildings
    BUILDING_SPEC[Terran.CommandCenter] = BuildingDetails("CommandCenter", 293, 18, 'C', 5)
    BUILDING_SPEC[Terran.SupplyDepot] = BuildingDetails("SupplyDepot", 81, 9, 'S', 4, SC2_Actions.BUILD_SUPPLY_DEPOT)
    BUILDING_SPEC[Terran.Barracks] = BuildingDetails("Barracks", 144, 12, 'B', 4, SC2_Actions.BUILD_BARRACKS)
    BUILDING_SPEC[Terran.Factory] = BuildingDetails("Factory", 144, 12, 'F', 4, SC2_Actions.BUILD_FACTORY)
    BUILDING_SPEC[Terran.Refinery] = BuildingDetails("OilRefinery", 144, 12, 'G', 4, SC2_Actions.BUILD_OIL_REFINERY)
    BUILDING_SPEC[Terran.BarracksReactor] = BuildingDetails("Reactor", 9, 3, 'R', 2, SC2_Actions.BUILD_REACTOR)
    BUILDING_SPEC[Terran.FactoryTechLab] = BuildingDetails("TechLab", 9, 3, 'T', 2, SC2_Actions.BUILD_TECHLAB)

    ARMY_SPEC = {}
    # army
    ARMY_SPEC[Terran.Marine] = UnitDetails("marine", 9, 1, 'm', SC2_Actions.TRAIN_MARINE)
    # ARMY_SPEC[56] = UnitDetails("raven", 12, 2)
    ARMY_SPEC[Terran.Reaper] = UnitDetails("reaper", 9, 1, 'r', SC2_Actions.TRAIN_REAPER)
    ARMY_SPEC[Terran.Marauder] = UnitDetails("marauder", 12, 2, 'a')
    ARMY_SPEC[Terran.Hellion] = UnitDetails("hellion", 12, 2, 'h', SC2_Actions.TRAIN_HELLION)
    ARMY_SPEC[Terran.SiegeTank] = UnitDetails("siege tank", 32, 3, 't', SC2_Actions.TRAIN_SIEGE_TANK)

    DEFAULT_UNIT_NUM_SCREEN_PIXELS = 9
    DEFAULT_BUILDING_NUM_SCREEN_PIXELS = 81

    SCV_SPEC = UnitDetails("scv", 9, 1, 's')

    # lut:
    UNIT_CHAR = {}
    for field in SC2_Params.NEUTRAL_MINERAL_FIELD[:]:
        UNIT_CHAR[field] = 'm'
    for gas in SC2_Params.VESPENE_GAS_FIELD[:]:
        UNIT_CHAR[gas] = 'g'
    for army in ARMY[:]:
        UNIT_CHAR[army] = 'a'
        

def GetUnitId(name):
    for uId, unit in TerranUnit.ARMY_SPEC.items():
        if unit.name == name:
            return uId
    
    return -1

# utils function
def Min(points):
    minVal = points[0]
    for i in range(1, len(points)):
        minVal = min(minVal, points[i])

    return minVal

def Max(points):
    maxVal = points[0]
    for i in range(1, len(points)):
        maxVal = max(maxVal, points[i])

    return maxVal

def DistForCmp(p1,p2):
    diffX = p1[SC2_Params.X_IDX] - p2[SC2_Params.X_IDX]
    diffY = p1[SC2_Params.Y_IDX] - p2[SC2_Params.Y_IDX]

    return diffX * diffX + diffY * diffY

def FindMiddle(points_y, points_x):
    min_x = Min(points_x)
    max_x = Max(points_x)
    midd_x = min_x + (max_x - min_x) / 2

    min_y = Min(points_y)
    max_y = Max(points_y)
    midd_y = min_y + (max_y - min_y) / 2

    return [int(midd_y), int(midd_x)]

def IsInScreen(y,x):
    return y >= 0 and y < SC2_Params.SCREEN_SIZE and x >= 0 and x < SC2_Params.SCREEN_SIZE

def Flood(location, buildingMap):   
    closeLocs = [[location[SC2_Params.Y_IDX] + 1, location[SC2_Params.X_IDX]], [location[SC2_Params.Y_IDX] - 1, location[SC2_Params.X_IDX]], [location[SC2_Params.Y_IDX], location[SC2_Params.X_IDX] + 1], [location[SC2_Params.Y_IDX], location[SC2_Params.X_IDX] - 1] ]
    points_y = [location[SC2_Params.Y_IDX]]
    points_x = [location[SC2_Params.X_IDX]]
    for loc in closeLocs[:]:
        if IsInScreen(loc[SC2_Params.Y_IDX],loc[SC2_Params.X_IDX]) and buildingMap[loc[SC2_Params.Y_IDX]][loc[SC2_Params.X_IDX]]:
            buildingMap[loc[SC2_Params.Y_IDX]][loc[SC2_Params.X_IDX]] = False
            pnts_y, pnts_x = Flood(loc, buildingMap)
            points_x.extend(pnts_x)
            points_y.extend(pnts_y)  

    return points_y, points_x


def IsolateArea(location, buildingMap):           
    return Flood(location, buildingMap)

def Scale2MiniMap(point, camNorthWestCorner, camSouthEastCorner):
    scaledPoint = [0,0]
    scaledPoint[SC2_Params.Y_IDX] = point[SC2_Params.Y_IDX] * (camSouthEastCorner[SC2_Params.Y_IDX] - camNorthWestCorner[SC2_Params.Y_IDX]) / SC2_Params.SCREEN_SIZE
    scaledPoint[SC2_Params.X_IDX] = point[SC2_Params.X_IDX] * (camSouthEastCorner[SC2_Params.X_IDX] - camNorthWestCorner[SC2_Params.X_IDX]) / SC2_Params.SCREEN_SIZE
    
    scaledPoint[SC2_Params.Y_IDX] += camNorthWestCorner[SC2_Params.Y_IDX]
    scaledPoint[SC2_Params.X_IDX] += camNorthWestCorner[SC2_Params.X_IDX]
    
    scaledPoint[SC2_Params.Y_IDX] = math.ceil(scaledPoint[SC2_Params.Y_IDX])
    scaledPoint[SC2_Params.X_IDX] = math.ceil(scaledPoint[SC2_Params.X_IDX])

    return scaledPoint

def Scale2Screen(point, camNorthWestCorner, camSouthEastCorner):
    scaledPoint = [0,0]
    scaledPoint[SC2_Params.Y_IDX] = point[SC2_Params.Y_IDX] - camNorthWestCorner[SC2_Params.Y_IDX]
    scaledPoint[SC2_Params.X_IDX] = point[SC2_Params.X_IDX] - camNorthWestCorner[SC2_Params.X_IDX]

    scaledPoint[SC2_Params.Y_IDX] = int(scaledPoint[SC2_Params.Y_IDX] * SC2_Params.SCREEN_SIZE / (camSouthEastCorner[SC2_Params.Y_IDX] - camNorthWestCorner[SC2_Params.Y_IDX]))
    scaledPoint[SC2_Params.X_IDX] = int(scaledPoint[SC2_Params.X_IDX] * SC2_Params.SCREEN_SIZE /  (camSouthEastCorner[SC2_Params.X_IDX] - camNorthWestCorner[SC2_Params.X_IDX]))

    return scaledPoint

def PowerSurroundPnt(point, radius2Include, powerMat):
    if radius2Include == 0:
        return powerMat[point[SC2_Params.Y_IDX]][point[SC2_Params.X_IDX]]

    power = 0
    for y in range(-radius2Include, radius2Include):
        for x in range(-radius2Include, radius2Include):
            power += powerMat[y + point[SC2_Params.Y_IDX]][x + point[SC2_Params.X_IDX]]

    return power

def BattleStarted(selfMat, enemyMat):
    attackRange = 1

    for xEnemy in range (attackRange, SC2_Params.MINIMAP_SIZE - attackRange):
        for yEnemy in range (attackRange, SC2_Params.MINIMAP_SIZE - attackRange):
            if enemyMat[yEnemy,xEnemy]:
                for xSelf in range(xEnemy - attackRange, xEnemy + attackRange):
                    for ySelf in range(yEnemy - attackRange, yEnemy + attackRange):
                        if enemyMat[ySelf][xSelf]:
                            return True, yEnemy, xEnemy

    return False, -1, -1

def PrintSpecificMat(mat, points = [], range2Include = 0, maxVal = -1):
    if maxVal == -1:
        maxVal = 0
        for vec in mat[:]:
            for val in vec[:]:
                maxVal = max(maxVal, val)

    toDivide = 1
    print("max val =", maxVal)
    if maxVal < 10:
        toAdd = ' '
    elif maxVal < 100:
        toAdd = '  '
        toAdd10 = ' '
    else:
        toAdd = '  '
        toAdd10 = ' '
        toDivide = 10

    for y in range(range2Include, SC2_Params.SCREEN_SIZE - range2Include):
        for x in range(range2Include, SC2_Params.SCREEN_SIZE - range2Include):
            prnted = False
            for i in range(0, len(points)):
                if x == points[i][SC2_Params.X_IDX] and y == points[i][SC2_Params.Y_IDX]:
                    print(" ", end = toAdd)
                    prnted = True
                    break
            if not prnted:
                sPower = PowerSurroundPnt([y,x], range2Include, mat)
                sPower = int(sPower / toDivide)
                if sPower < 10:
                    print(sPower, end = toAdd)
                elif sPower < 100:
                    print(sPower, end = toAdd10)
        print('|')
    
    print("\n")



def SwapPnt(point):
    return point[1], point[0]

def GetCoord(idxLocation, gridSize_x):
    ret = [-1,-1]
    ret[SC2_Params.Y_IDX] = int(idxLocation / gridSize_x)
    ret[SC2_Params.X_IDX] = idxLocation % gridSize_x
    return ret

def PrintSingleBuildingSize(buildingMap, name):
    allPnts_y, allPnts_x = buildingMap.nonzero()
    if len(allPnts_y > 0):
        pnts_y, pnts_x = IsolateArea([allPnts_y[0], allPnts_x[0]], buildingMap)
        size_y = Max(pnts_y) - Min(pnts_y)
        size_x = Max(pnts_x) - Min(pnts_x)
        print(name , "size x = ", size_x, "size y = ", size_y)

def PrintBuildingSizes(unit_type):
    ccMap = unit_type == Terran.CommandCenter
    PrintSingleBuildingSize(ccMap, "command center")
    sdMap = unit_type == Terran.SupplyDepot
    PrintSingleBuildingSize(sdMap, "supply depot")
    baMap = unit_type == Terran.Barracks
    PrintSingleBuildingSize(baMap, "barracks")

def GetScreenCorners(obs):
    cameraLoc = obs.observation['feature_minimap'][SC2_Params.CAMERA]
    ca_y, ca_x = cameraLoc.nonzero()

    return [ca_y.min(), ca_x.min()] , [ca_y.max(), ca_x.max()]

def BlockingType(unit):
    return TerranUnit.BLOCKING_TYPE[unit]


def HaveSpaceMiniMap(occupyMat, heightsMap, yStart, xStart, neededSize):
    height = heightsMap[yStart][xStart]
    if height == 0:
        return False

    yEnd = min(yStart + neededSize, SC2_Params.MINIMAP_SIZE)
    xEnd = min(xStart + neededSize, SC2_Params.MINIMAP_SIZE)
    for y in range (yStart, yEnd):
        for x in range (xStart, xEnd):
            if occupyMat[y][x] or height != heightsMap[y][x]:
                return False
    
    return True

def PrintMiniMap(obs, cameraCornerNorthWest, cameraCornerSouthEast):
    selfPnt_y, selfPnt_x = (obs.observation['feature_minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
    enemyPnt_y, enemyPnt_x = (obs.observation['feature_minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE).nonzero()

    for y in range(SC2_Params.MINIMAP_SIZE):
        for x in range(SC2_Params.MINIMAP_SIZE):
            isSelf = False
            for i in range (0, len(selfPnt_y)):
                if (y == selfPnt_y[i] and x == selfPnt_x[i]):
                    isSelf = True
            
            isEnemy = False
            for i in range (0, len(enemyPnt_y)):
                if (y == enemyPnt_y[i] and x == enemyPnt_x[i]):
                    isEnemy = True

            if (x == cameraCornerNorthWest[SC2_Params.X_IDX] and y == cameraCornerNorthWest[SC2_Params.Y_IDX]) or (x == cameraCornerSouthEast[SC2_Params.X_IDX] and y == cameraCornerSouthEast[SC2_Params.Y_IDX]):
                print ('#', end = '')
            elif isSelf:
                print ('s', end = '')
            elif isEnemy:
                print ('e', end = '')
            else:
                print ('_', end = '')
        print('|')  

def PrintScreen(unitType, addPoints = [], valToPrint = -1):
    nonPrintedVals = []
    for y in range(0, SC2_Params.SCREEN_SIZE):
        for x in range(0, SC2_Params.SCREEN_SIZE):        
            foundInPnts = False
            for i in range(0, len (addPoints)):
                if addPoints[i][SC2_Params.X_IDX] == x and addPoints[i][SC2_Params.Y_IDX] == y:
                    foundInPnts = True

            uType = unitType[y][x]
            if foundInPnts:
                print (' ', end = '')
            elif uType == valToPrint:
                print ('V', end = '')
            elif uType in TerranUnit.BUILDING_SPEC.keys():
                print(TerranUnit.BUILDING_SPEC[uType].name, end = '')
            elif uType in TerranUnit.ARMY_SPEC.keys():
                print(TerranUnit.ARMY_SPEC[uType].name, end = '')
            elif uType in TerranUnit.UNIT_CHAR:
                print(TerranUnit.UNIT_CHAR[uType], end = '')
            else:
                if uType not in nonPrintedVals:
                    nonPrintedVals.append(uType)
        print('|') 

    if len(nonPrintedVals) > 0:
        print("non printed vals = ", nonPrintedVals) 
        time.sleep(1)
        SearchNewBuildingPnt(unitType)

def SearchNewBuildingPnt(unitType):
    print("search new building point")
    for i in range(1, 100):
        if i not in TerranUnit.UNIT_CHAR:
            pnts_y,pnts_x = (unitType == i).nonzero()
            if len(pnts_y) > 0:
                PrintScreen(unitType, [], i)
                print("exist idx =", i, "\n\n\n")


def GetLocationForBuildingMiniMap(obs, commandCenterLoc, buildingType):
    if buildingType == Terran.Refinery:
        return [-1,-1]

    height_map = obs.observation['feature_minimap'][SC2_Params.HEIGHT_MAP]
    occupyMat = obs.observation['feature_minimap'][SC2_Params.PLAYER_RELATIVE_MINIMAP] > 0
    neededSize = TerranUnit.BUILDING_SPEC[buildingType].miniMapSize

    location = [-1, -1]
    minDist = SC2_Params.MAX_MINIMAP_DIST
    for y in range(0, SC2_Params.MINIMAP_SIZE - neededSize):
        for x in range(0, SC2_Params.MINIMAP_SIZE - neededSize):
            foundLoc = HaveSpaceMiniMap(occupyMat, height_map, y, x, neededSize)       
            if foundLoc:
                currLocation = [y + int(neededSize / 2), x + int(neededSize / 2)]
                currDist = DistForCmp(currLocation, commandCenterLoc)
                if currDist < minDist:
                    location = currLocation
                    minDist = currDist



    return location

def BlockingResourceGather(unitType, y, x, ccMat, nonAllowedDirections2CC, maxCC):
    if x > maxCC[SC2_Params.X_IDX] or y > maxCC[SC2_Params.Y_IDX]:
        return False

    jump = 5
    for direction in nonAllowedDirections2CC:
        x2Check = x + direction[SC2_Params.X_IDX] * jump
        y2Check = y + direction[SC2_Params.Y_IDX] * jump
        if ccMat[y2Check][x2Check]:
            return True
    
    return False

def GetLocationForBuilding(obs, buildingType, notAllowedDirections2CC = [], nonValidCoord = [], additionType = None):
    unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
    if buildingType == Terran.Refinery:
        return GetLocationForOilRefinery(unitType)

    ccMat = unitType == Terran.CommandCenter
    hasCC = ccMat.any()
    if hasCC:
        maxCC = np.max(ccMat.nonzero(), axis=1)

    neededSizeY = TerranUnit.BUILDING_SPEC[buildingType].screenPixels1Axis
    neededSizeX = neededSizeY
    if additionType != None:
        neededSizeX = neededSizeX + TerranUnit.BUILDING_SPEC[additionType].screenPixels1Axis

    cameraHeightMap = obs.observation['feature_screen'][SC2_Params.HEIGHT_MAP]

    foundLoc = False
    location = [-1, -1]
    freeMat = np.zeros((SC2_Params.SCREEN_SIZE, SC2_Params.SCREEN_SIZE), int)
    for y in range(SC2_Params.SCREEN_SIZE):
        for x in range(SC2_Params.SCREEN_SIZE): 
            if TerranUnit.BLOCKING_TYPE[unitType[y][x]] or InNonValidCoord(y, x, nonValidCoord):
                freeMat[y,x] = -1
            elif hasCC and BlockingResourceGather(unitType, y, x, ccMat, notAllowedDirections2CC, maxCC):
                freeMat[y,x] = -1
            else:

                freeMat[y,x] = cameraHeightMap[y][x]

            yStart = y - neededSizeY + 1
            xStart = x - neededSizeX + 1
            
            if yStart >= 0 and xStart >= 0:
                yEnd = yStart + neededSizeY
                xEnd = xStart + neededSizeX
                heightVal = cameraHeightMap[yStart][xStart]
                if heightVal > 0:
                    foundLoc = (freeMat[yStart:yEnd, xStart:xEnd] == heightVal).all()
                    if foundLoc:
                        location = [yStart + int(neededSizeY / 2), xStart + int(neededSizeX / 2)]
                        break
                        
        if foundLoc:
            break

    return location

def InNonValidCoord(y, x, nonValidCoord):
    for c in nonValidCoord:
        if c[SC2_Params.X_IDX] == x and c[SC2_Params.Y_IDX] == y:
            return True

    return False

def GetLocationForOilRefinery(unitType):
    refMat = unitType == Terran.Refinery
    ref_y,ref_x = refMat.nonzero()
    gasMat = unitType == SC2_Params.VESPENE_GAS_FIELD
    vg_y, vg_x = gasMat.nonzero()


    if len(vg_y) == 0:
        return [-1, -1]
    
    if len(ref_y) == 0:
        # no refineries
        location = vg_y[0], vg_x[0]
        vg_y, vg_x = IsolateArea(location, gasMat)
        midPnt = FindMiddle(vg_y, vg_x)
        return midPnt
    else:
        rad2Include = 4

        initLoc = False
        for pnt in range(0, len(vg_y)):
            found = False
            i = 0
            while not found and i < len(ref_y):
                if abs(ref_y[i] - vg_y[pnt]) < rad2Include and abs(ref_x[i] - vg_x[pnt]) < rad2Include:
                    found = True
                i += 1

            if not found:
                initLoc = True
                location = vg_y[pnt], vg_x[pnt]
                break
        
        if initLoc:
            newVG_y, newVG_x = IsolateArea(location, gasMat)
            midPnt = FindMiddle(newVG_y, newVG_x)
            return midPnt

    return [-1, -1]

def FindBuildingRightEdge(unitType, buildingType, point):
    buildingMat = unitType == buildingType
    found = False
    x = point[SC2_Params.X_IDX]
    y = point[SC2_Params.Y_IDX]

    while not found:
        if x + 1 >= SC2_Params.SCREEN_SIZE:
            break 

        x += 1
        if not buildingMat[y][x]:
            if y + 1 < SC2_Params.SCREEN_SIZE and buildingMat[y + 1][x]:
                y += 1
            elif y > 0 and buildingMat[y - 1][x]:
                y -= 1
            else:
                found = True

    return y,x

def CenterPoints(y, x, numPixels1Unit = 1):
    groups = Grouping(y, x)
    def Avg(group):
        sumG = [0,0]
        for p in group:
            sumG[0] += p[0]
            sumG[1] += p[1]

        return [int(sumG[0] / len(group)), int(sumG[1] / len(group))]
    
    points= []
    groupSizes = []
    for g in groups:
        pnt = Avg(g)
        points.append(pnt)
        groupSizes.append(math.ceil(len(g) / numPixels1Unit))

    
    return points, groupSizes

def Grouping(y, x):

    groups = []  
    def inGroup(groups, pnt):
        for g in groups:
            if pnt in g:
                return True
        return False

    def isNearGroup(g, pnt):
        for p in g:
            diff = abs(p[0] - pnt[0]) + abs(p[1] - pnt[1])
            if diff == 1:
                return True
        return False

    def nearGroup(groups, pnt):
        for i in range(len(groups)):
            if isNearGroup(groups[i],pnt):
                return i
        return -1
        
    for i in range(0, len(y)):
        pnt = [y[i], x[i]]
        if not inGroup(groups, pnt):
            g = nearGroup(groups, pnt)
            if g >= 0 :
                groups[g].append(pnt)
            else:
                groups.append([pnt])

    def toJoin(g1, g2):
        for p in g1:
            if isNearGroup(g2,p):
                return True
        return False

    joinedGroup = True
    while joinedGroup:
        toRemove = []
        for i in range(len(groups)):
            for j in range(i + 1,len(groups)):
                if toJoin(groups[i], groups[j]):
                    groups[i] += groups[j]
                    toRemove.append(j)

        
        joinedGroup = len(toRemove) > 0
        newGroups = []
        for i in range(len(groups)):
            if i not in toRemove:
                newGroups.append(groups[i].copy())
        groups = newGroups

    return groups

neighbors2CheckBuilding = []

neighbors2CheckBuilding.append([-1,-2])
neighbors2CheckBuilding.append([-2,-1])

neighbors2CheckUnit = []
neighbors2CheckUnit.append([-1, 0])
neighbors2CheckUnit.append([1, 0])
neighbors2CheckUnit.append([0, 1])
neighbors2CheckUnit.append([0, -1])

def SelectBuildingValidPoint(unit_type, buildingType):
    buildingMap = unit_type == buildingType
    y, x = buildingMap.nonzero()

    if len(y) == 0:
        return [-1,-1]
    idx = 0
    for i in range(len(y)):
        if IsValidPoint4Select(buildingMap, y[i], x[i]):
            idx = i
            break

    return y[idx], x[idx]


def SelectUnitValidPoints(unitMap):
    p_y, p_x = (unitMap).nonzero()
    valid_y = []
    valid_x = []
    for i in range(len(p_y)):
        x = p_x[i]
        y = p_y[i]
        if IsValidPoint4Select(unitMap, y, x, neighbors2CheckUnit):
            valid_y.append(y)
            valid_x.append(x)

    return valid_y, valid_x

def IsValidPoint4Select(buildingMap, y, x, neighbor2Check = neighbors2CheckBuilding):
    if x <= 1 or y <= 1 or x >= SC2_Params.SCREEN_SIZE - 1 or y >= SC2_Params.SCREEN_SIZE - 1:
        return False

    for neighbor in neighbor2Check:
        if not buildingMap[y + neighbor[0]][x + neighbor[1]]:
            return False

    return True 

def GatherResource(unitType, resourceList):
    allResMat = np.in1d(unitType, resourceList).reshape(unitType.shape)
    unit_y, unit_x = SelectUnitValidPoints(allResMat)
    if len(unit_y) > 0:
        i = random.randint(0, len(unit_y) - 1)
        return [unit_y[i], unit_x[i]]
    
    return [-1,-1]

def GetSelectedUnits(obs):
    scvStatus = list(obs.observation['multi_select'])
    if len(scvStatus) ==  0:
        scvStatus = list(obs.observation['single_select'])
    return scvStatus


def SupplyCap(buildingCompleted):
    return 15 * len(buildingCompleted[Terran.CommandCenter]) + 8 * len(buildingCompleted[Terran.SupplyDepot])