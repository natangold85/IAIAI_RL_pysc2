import random
import math
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

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
    BUILD_REACTOR = actions.FUNCTIONS.Build_Reactor_screen.id
    BUILD_TECHLAB = actions.FUNCTIONS.Build_TechLab_screen.id

    # train army action
    TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
    TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

    TRAIN_HELLION = actions.FUNCTIONS.Train_Hellion_quick.id
    TRAIN_SIEGE_TANK = actions.FUNCTIONS.Train_SiegeTank_quick.id

    SELECT_ARMY = actions.FUNCTIONS.select_army.id
    MOVE_IN_SCREEN = actions.FUNCTIONS.Move_screen.id
    ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
    ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id

    



class SC2_Params:
    # minimap feature
    CAMERA = features.MINIMAP_FEATURES.camera.index
    HEIGHT_MINIMAP = features.MINIMAP_FEATURES.height_map.index
    PLAYER_RELATIVE_MINIMAP = features.MINIMAP_FEATURES.player_relative.index
    
    # screen feature
    HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
    PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
    UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
    PLAYER_ID = features.SCREEN_FEATURES.player_id.index
    SELECTED_IN_SCREEN = features.SCREEN_FEATURES.selected.index
    
    PLAYER_SELF = 1
    PLAYER_HOSTILE = 4
    ARMY_SUPPLY = 5

    #player general info
    MINERALS = 1
    VESPENE = 2
    SUPPLY_CAP = 4
    IDLE_WORKER_COUNT = 7

    # single and multi select table idx
    BUILDING_COMPLETION_IDX = 6

    # multi and single select information
    NOT_QUEUED = [0]
    QUEUED = [1]
    SELECT_SINGLE = [0]
    SELECT_ALL = [2]

    NEUTRAL_MINERAL_FIELD = [341, 483]
    VESPENE_GAS_FIELD = [342]

    Y_IDX = 0
    X_IDX = 1

    MINIMAP_SIZE = [64, 64]
    MAX_MINIMAP_DIST = MINIMAP_SIZE[X_IDX] * MINIMAP_SIZE[X_IDX] + MINIMAP_SIZE[Y_IDX] * MINIMAP_SIZE[Y_IDX] 
    MINI_MAP_START = [8, 8]
    MINI_MAP_END = [61, 50]

    SCREEN_SIZE = [84, 84]

    TOPLEFT_BASE_LOCATION = [23,18]
    BOTTOMRIGHT_BASE_LOCATION = [45,39]


class TerranUnit:
    COMMANDCENTER = 18
    SCV = 45 
    SUPPLY_DEPOT = 19
    OIL_REFINERY = 20
    BARRACKS = 21
    FACTORY = 27
    REACTOR = 38
    TECHLAB = 39

    ARMY= [53,40,48,49,33]

    FLYING_BARRACKS = 46
    FLYING_FACTORY = 43

    # army specific:
    MARINE = 48
    MARINE_SCREEN_SIZE_ONE_AXIS = 3
    MARINE_SCREEN_NUM_PIXELS = 9
    # lut:

    BUILDING_NAMES = {}
    BUILDING_SIZES = {}
    BUILDING_MINIMAP_SIZES = {}

    BUIILDING_2_SC2ACTIONS = {}
    UNIT_CHAR = {}

    BUILDING_NAMES[COMMANDCENTER] = "CommandCenter"
    BUILDING_NAMES[SUPPLY_DEPOT] = "SupplyDepot"
    BUILDING_NAMES[BARRACKS] = "Barracks"
    BUILDING_NAMES[FACTORY] = "Factory"
    BUILDING_NAMES[OIL_REFINERY] = "OilRefinery"

    BUILDING_NAMES[REACTOR] = "Reactor"
    BUILDING_NAMES[TECHLAB] = "TechLab"

    BUILDING_SIZES[COMMANDCENTER] = 18
    BUILDING_SIZES[SUPPLY_DEPOT] = 9
    BUILDING_SIZES[BARRACKS] = 12
    BUILDING_SIZES[FACTORY] = 12

    BUILDING_SIZES[REACTOR] = 3
    BUILDING_SIZES[TECHLAB] = 3

    BUILDING_MINIMAP_SIZES[COMMANDCENTER] = 5
    BUILDING_MINIMAP_SIZES[SUPPLY_DEPOT] = 4
    BUILDING_MINIMAP_SIZES[BARRACKS] = 4
    BUILDING_MINIMAP_SIZES[FACTORY] = 4

    BUILDING_MINIMAP_SIZES[REACTOR] = 5
    BUILDING_MINIMAP_SIZES[TECHLAB] = 5

    BUIILDING_2_SC2ACTIONS[OIL_REFINERY] = SC2_Actions.BUILD_OIL_REFINERY
    BUIILDING_2_SC2ACTIONS[SUPPLY_DEPOT] = SC2_Actions.BUILD_SUPPLY_DEPOT
    BUIILDING_2_SC2ACTIONS[BARRACKS] = SC2_Actions.BUILD_BARRACKS
    BUIILDING_2_SC2ACTIONS[FACTORY] = SC2_Actions.BUILD_FACTORY

    UNIT_CHAR[0] = '_'
    UNIT_CHAR[COMMANDCENTER] = 'C'
    UNIT_CHAR[SCV] = 's'
    UNIT_CHAR[SUPPLY_DEPOT] = 'S'
    UNIT_CHAR[OIL_REFINERY] = 'G'
    UNIT_CHAR[BARRACKS] = 'B'
    UNIT_CHAR[FACTORY] = 'F'
    UNIT_CHAR[REACTOR] = 'R'
    UNIT_CHAR[TECHLAB] = 'T'

    UNIT_CHAR[FLYING_BARRACKS] = 'Y'
    UNIT_CHAR[FLYING_FACTORY] = 'Y'

    DO_NOTHING_BUILDING_CHECK = [COMMANDCENTER, SUPPLY_DEPOT, OIL_REFINERY, BARRACKS, FACTORY]
    for field in SC2_Params.NEUTRAL_MINERAL_FIELD[:]:
        UNIT_CHAR[field] = 'm'
    for gas in SC2_Params.VESPENE_GAS_FIELD[:]:
        UNIT_CHAR[gas] = 'g'
    for army in ARMY[:]:
        UNIT_CHAR[army] = 'a'
        

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
    return y >= 0 and y < SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] and x >= 0 and x < SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX] 

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
    scaledPoint[SC2_Params.Y_IDX] = point[SC2_Params.Y_IDX] * (camSouthEastCorner[SC2_Params.Y_IDX] - camNorthWestCorner[SC2_Params.Y_IDX]) / SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] 
    scaledPoint[SC2_Params.X_IDX] = point[SC2_Params.X_IDX] * (camSouthEastCorner[SC2_Params.X_IDX] - camNorthWestCorner[SC2_Params.X_IDX]) / SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX] 
    
    scaledPoint[SC2_Params.Y_IDX] += camNorthWestCorner[SC2_Params.Y_IDX]
    scaledPoint[SC2_Params.X_IDX] += camNorthWestCorner[SC2_Params.X_IDX]
    
    scaledPoint[SC2_Params.Y_IDX] = math.ceil(scaledPoint[SC2_Params.Y_IDX])
    scaledPoint[SC2_Params.X_IDX] = math.ceil(scaledPoint[SC2_Params.X_IDX])

    return scaledPoint

def Scale2Screen(point, camNorthWestCorner, camSouthEastCorner):
    scaledPoint = [0,0]
    scaledPoint[SC2_Params.Y_IDX] = point[SC2_Params.Y_IDX] - camNorthWestCorner[SC2_Params.Y_IDX]
    scaledPoint[SC2_Params.X_IDX] = point[SC2_Params.X_IDX] - camNorthWestCorner[SC2_Params.X_IDX]

    scaledPoint[SC2_Params.Y_IDX] = int(scaledPoint[SC2_Params.Y_IDX] * SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] / (camSouthEastCorner[SC2_Params.Y_IDX] - camNorthWestCorner[SC2_Params.Y_IDX]))
    scaledPoint[SC2_Params.X_IDX] = int(scaledPoint[SC2_Params.X_IDX] * SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX] /  (camSouthEastCorner[SC2_Params.X_IDX] - camNorthWestCorner[SC2_Params.X_IDX]))

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

    for xEnemy in range (attackRange, SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX] - attackRange):
        for yEnemy in range (attackRange, SC2_Params.MINIMAP_SIZE[SC2_Params.Y_IDX] - attackRange):
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

    for y in range(range2Include, SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] - range2Include):
        for x in range(range2Include, SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX] - range2Include):
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
    ccMap = unit_type == TerranUnit.COMMANDCENTER
    PrintSingleBuildingSize(ccMap, "command center")
    sdMap = unit_type == TerranUnit.SUPPLY_DEPOT
    PrintSingleBuildingSize(sdMap, "supply depot")
    baMap = unit_type == TerranUnit.BARRACKS
    PrintSingleBuildingSize(baMap, "barracks")

def GetScreenCorners(obs):
    cameraLoc = obs.observation['minimap'][SC2_Params.CAMERA]
    ca_y, ca_x = cameraLoc.nonzero()

    return [ca_y.min(), ca_x.min()] , [ca_y.max(), ca_x.max()]

def BlockingType(unit):
    return unit > 0 and unit != TerranUnit.SCV and unit not in TerranUnit.ARMY

def HaveSpace(unitType, heightsMap, yStart, xStart, neededSize):
    height = heightsMap[yStart][xStart]
    if height == 0:
        return False

    yEnd = min(yStart + neededSize, SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX])
    xEnd = min(xStart + neededSize, SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX])
    for y in range (yStart, yEnd):
        for x in range (xStart, xEnd):
            if BlockingType(unitType[y][x]) or height != heightsMap[y][x]:
                return False
    
    return True

def HaveSpaceMiniMap(occupyMat, heightsMap, yStart, xStart, neededSize):
    height = heightsMap[yStart][xStart]
    if height == 0:
        return False

    yEnd = min(yStart + neededSize, SC2_Params.MINIMAP_SIZE[SC2_Params.Y_IDX])
    xEnd = min(xStart + neededSize, SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX])
    for y in range (yStart, yEnd):
        for x in range (xStart, xEnd):
            if occupyMat[y][x] or height != heightsMap[y][x]:
                return False
    
    return True

def PrintMiniMap(obs, cameraCornerNorthWest, cameraCornerSouthEast):
    selfPnt_y, selfPnt_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
    enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE).nonzero()

    for y in range(SC2_Params.MINI_MAP_START[SC2_Params.Y_IDX], SC2_Params.MINI_MAP_END[SC2_Params.Y_IDX]):
        for x in range(SC2_Params.MINI_MAP_START[SC2_Params.X_IDX], SC2_Params.MINI_MAP_END[SC2_Params.X_IDX]):
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
    for y in range(0, SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX]):
        for x in range(0, SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX]):        
            foundInPnts = False
            for i in range(0, len (addPoints)):
                if addPoints[i][SC2_Params.X_IDX] == x and addPoints[i][SC2_Params.Y_IDX] == y:
                    foundInPnts = True

            uType = unitType[y][x]
            if foundInPnts:
                print (' ', end = '')
            elif uType == valToPrint:
                print ('V', end = '')
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
    if buildingType == TerranUnit.OIL_REFINERY:
        return [-1,-1]

    height_map = obs.observation['minimap'][SC2_Params.HEIGHT_MAP]
    occupyMat = obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE_MINIMAP] > 0
    neededSize = TerranUnit.BUILDING_MINIMAP_SIZES[buildingType]

    location = [-1, -1]
    minDist = SC2_Params.MAX_MINIMAP_DIST
    for y in range(0, SC2_Params.MINIMAP_SIZE[SC2_Params.Y_IDX] - neededSize):
        for x in range(0, SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX] - neededSize):
            foundLoc = HaveSpaceMiniMap(occupyMat, height_map, y, x, neededSize)       
            if foundLoc:
                currLocation = [y + int(neededSize / 2), x + int(neededSize / 2)]
                currDist = DistForCmp(currLocation, commandCenterLoc)
                if currDist < minDist:
                    location = currLocation
                    minDist = currDist



    return location

def BlockingResourceGather(unitType, y, x, neededSize):

    cc_y, cc_x = (unitType == TerranUnit.COMMANDCENTER).nonzero()
    if len (cc_y) == 0:
        return False

    for yB in range (y, y +neededSize):
        for xB in range (x, x + neededSize):

            minDist = 100000
            minIdx = -1
            for i in range(0, len(cc_y)):
                dist = DistForCmp([cc_y[i], cc_x[i]], [yB, xB])
                if dist < minDist:
                    minDist = dist
                    minIdx= i

            xDiff = xB - cc_x[minIdx]
            yDiff = yB - cc_y[minIdx]

            if yDiff == 0:
                changeY = 0
                if xDiff > 0:
                    changeX = 1
                else:
                    changeX = -1 
            elif xDiff == 0:
                changeX = 0
                if yDiff > 0:
                    changeY = 1
                else:
                    changeY = -1 
            else:
                slope = yDiff / xDiff   
                if abs(slope) > 1:
                    changeX = xDiff / yDiff
                    if y > cc_y[minIdx]:
                        changeY = 1
                    else:
                        changeY = -1
                
                else:
                    changeY = slope
                    if x > cc_x[minIdx]:
                        changeX = 1
                    else:
                        changeX = -1

            currX = x
            currY = y
            for i in range(0, 10):
                currX += changeX
                currY += changeY
                intX = int(currX)
                intY = int(currY)

                if IsInScreen(intY, intX):
                    unit = TerranUnit.UNIT_CHAR[unitType[intY][intX]]
                    if unit == 'm' or unit == 'g' or unit == 'G':
                        return True
                else:
                    break
     
    return False

def GetLocationForBuilding(obs, cameraCornerNorthWest, cameraCornerSouthEast, buildingType):
    unitType = obs.observation['screen'][SC2_Params.UNIT_TYPE]
    if buildingType == TerranUnit.OIL_REFINERY:
        return GetLocationForOilRefinery(unitType)

    neededSize = TerranUnit.BUILDING_SIZES[buildingType]
    cameraHeightMap = obs.observation['screen'][SC2_Params.HEIGHT_MAP]

    foundLoc = False
    location = [-1, -1]
    for y in range(0, SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] - neededSize):
        for x in range(0, SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX] - neededSize):                
            toPrint = False
            if HaveSpace(unitType, cameraHeightMap, y, x, neededSize) and not BlockingResourceGather(unitType, y, x, neededSize):
                foundLoc = True
                location = [y + int(neededSize / 2), x + int(neededSize / 2)]
                break

        if foundLoc:
            break

    return location

def GetLocationForOilRefinery(unitType):
    refMat = unitType == TerranUnit.OIL_REFINERY
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

def GetLocationForBuildingAddition(obs, buildingType, camNorthWest, camSouthEast, defaultPnt = []):
    neededSize = TerranUnit.BUILDING_SIZES[buildingType]
    additionSize = TerranUnit.BUILDING_SIZES[TerranUnit.REACTOR]
    unitType = obs.observation['screen'][SC2_Params.UNIT_TYPE]
    
    cameraHeightMap = obs.observation['screen'][SC2_Params.HEIGHT_MAP]
  
    # find right edge of building
    if len(defaultPnt) > 0:
        y, x = FindBuildingRightEdge(unitType, buildingType, defaultPnt)
        if y < SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] and x < SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX] and HaveSpace(unitType, cameraHeightMap, y, x, additionSize):
            return defaultPnt

    foundLoc = False
    location = [-1, -1]
    for y in range(0, SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] - neededSize):
        for x in range(0, SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX] - neededSize - additionSize):
            if HaveSpace(unitType, cameraHeightMap, y, x, neededSize):
                additionY = y + int((neededSize / 2) - (additionSize / 2))
                additionX = x + neededSize
                foundLoc = HaveSpace(unitType, cameraHeightMap, additionY, additionX, additionSize)
                
            if foundLoc:
                location = [y + int(neededSize / 2), x + int(neededSize / 2)]
                break

        if foundLoc:
            break

    return location

def FindBuildingRightEdge(unitType, buildingType, point):
    buildingMat = unitType == buildingType
    found = False
    x = point[SC2_Params.X_IDX]
    y = point[SC2_Params.Y_IDX]

    while not found:
        if x + 1 >= SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX]:
            break 

        x += 1
        if not buildingMat[y][x]:
            if y + 1 < SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] and buildingMat[y + 1][x]:
                y += 1
            elif y > 0 and buildingMat[y - 1][x]:
                y -= 1
            else:
                found = True

    return y,x



import numpy as np
import pandas as pd
import os.path

class QTableParamsWOChangeInExploration:
    def __init__(self, learning_rate=0.01, reward_decay=0.9, explorationProb=0.1):
        self.isExplorationDecay = False
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.explorationProb = explorationProb

class QTableParamsWithChangeInExploration:
    def __init__(self, exploreRate = 0.0001, exploreStart = 1, exploreStop = 0.01, learning_rate=0.01, reward_decay=0.9):
        self.isExplorationDecay = True
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.exploreStart = exploreStart
        self.exploreStop = exploreStop
        self.exploreRate = exploreRate


class TableMngr:
    def __init__(self, numActions, qTableName, QTableParams, transitionTableName = '', resultFileName = '', rewardTableName = '', numToWriteResult = 100):
        self.qTable = QLearningTable(numActions, qTableName, QTableParams)

        if transitionTableName != '':
            self.createTransitionTable = True
            self.tTable = TransitionTable(numActions, transitionTableName)
        else:
            self.createTransitionTable = False

        if rewardTableName != '':
            self.createRewardTable = True
            self.rTable = RewardTable(rewardTableName)
        else:
            self.createRewardTable = False

        if resultFileName != '':
            self.createResultFile = True
            self.resultFile = ResultFile(resultFileName, numToWriteResult)
        else:
            self.createResultFile = False

    def choose_action(self, observation):
        return self.qTable.choose_action(observation)

    def learn(self, s, a, r, s_):
        self.qTable.learn(s, a, r, s_)

        if self.createTransitionTable:
            self.tTable.learn(s, a, s_)

        if self.createRewardTable:
            self.rTable.learn(s_, r)

    def end_run(self, s, a, r):
        self.qTable.learn(s, a, r, 'terminal')
        self.qTable.end_run(r)

        if r > 0:
            s_ = 'win'
        elif r < 0:
            s_ = 'loss'
        else:
            s_ = 'tie'

        if self.createTransitionTable:
            self.tTable.learn(s, a, s_)  
            self.tTable.end_run()

        if self.createRewardTable:
            self.rTable.learn(s_, r)
            self.rTable.end_run()

        if self.createResultFile:
            self.resultFile.end_run(r)

class QLearningTable:
    def __init__(self, numActions, qTableName, qTableParams):
        self.qTableName = qTableName
        self.actions = list(range(numActions))  # a list
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float)
        if os.path.isfile(qTableName + '.gz'):
            self.q_table = pd.read_pickle(qTableName + '.gz', compression='gzip')
        
        self.params = qTableParams
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        self.NumRunsExperimentSlot = 1
        
        self.AvgRewardSlot = 2
        self.AvgRewardExperimentSlot = 3

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot]
        self.avgTotReward = self.q_table.ix[self.TrialsData, self.AvgRewardSlot]
        self.numExpRuns = 0
        self.avgExpReward = 0

        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = 0
        self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot] = 0

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if self.params.isExplorationDecay:
            exploreProb = self.params.exploreStop + (self.params.exploreStart - self.params.exploreStop) * np.exp(-self.params.exploreRate * self.numTotRuns)
        else:
            exploreProb = self.params.explorationProb

        if np.random.uniform() > exploreProb:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s)
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal':
            self.check_state_exist(s_)
            q_target = r + self.params.rewardDecay * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        
        # update
        self.q_table.ix[s, a] += self.params.learningRate * (q_target - q_predict)

    def end_run(self, r):
    
        self.avgTotReward = (self.numTotRuns * self.avgTotReward + r) / (self.numTotRuns + 1)
        self.avgExpReward = (self.numExpRuns * self.avgExpReward + r) / (self.numExpRuns + 1)
        
        self.numTotRuns += 1
        self.numExpRuns += 1

        self.q_table.ix[self.TrialsData, self.AvgRewardSlot] = self.avgTotReward
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = self.avgExpReward

        
        self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns
        self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot] = self.numExpRuns

        print("num total runs = ", self.numTotRuns, "avg total = ", self.avgTotReward)
        print("num experiment runs = ", self.numExpRuns, "avg experiment = ", self.avgExpReward)

        self.q_table.to_pickle(self.qTableName + '.gz', 'gzip') 

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class TransitionTable:
    def __init__(self, numActions, tableName):
        self.tableName = tableName
        self.actions = list(range(numActions))  # a list
        self.t_table = pd.DataFrame(columns=self.actions, dtype=np.float)
        if os.path.isfile(tableName + '.gz'):
            self.t_table = pd.read_pickle(tableName + '.gz', compression='gzip')
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.t_table.ix[self.TrialsData, self.NumRunsTotalSlot]

    def check_state_exist(self, state):
        if state not in self.t_table.index:
            # append new state to q table
            self.t_table = self.t_table.append(pd.Series([0] * len(self.actions), index=self.t_table.columns, name=state))

    def learn(self, s, a, s_):
        state = s + "__" + s_
        self.check_state_exist(state)             
        # update transition
        self.t_table.ix[state, a] += 1

    def end_run(self):
   
        self.numTotRuns += 1      
        self.t_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns

        self.t_table.to_pickle(self.tableName + '.gz', 'gzip') 

class RewardTable:
    def __init__(self, tableName):

        self.tableName = tableName
        self.rewardIdx = 0
        self.rewardCol = list(range(1))

        self.r_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(tableName + '.gz'):
            self.r_table = pd.read_pickle(tableName + '.gz', compression='gzip')
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.r_table.ix[self.TrialsData, self.NumRunsTotalSlot]

    def check_state_exist(self, state):
        if state not in self.r_table.index:
            # append new state to q table
            self.r_table = self.r_table.append(pd.Series([0] * len(self.rewardCol), index=self.r_table.columns, name=state))
            return True

        return False

    def learn(self, s_, r):
        if r != 0:
            if self.check_state_exist(s_):
                # insert values
                self.r_table.ix[s_, self.rewardIdx] = r
        
    def end_run(self):
   
        self.numTotRuns += 1      
        self.r_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns
        self.r_table.to_pickle(self.tableName + '.gz', 'gzip') 

class ResultFile:
    def __init__(self, tableName, numToWrite = 100):
        self.tableName = tableName
        self.numToWrite = numToWrite

        # keys
        self.inMiddleValKey = 'middleCalculationVal'
        self.inMiddleCountKey = 'middleCalculationCount'
        self.countCompleteKey = 'countComplete'
        self.prevNumToWriteKey = 'prevNumToWrite'

        self.rewardCol = list(range(1))
        self.result_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(tableName + '.gz'):
            self.result_table = pd.read_pickle(tableName + '.gz', compression='gzip')

        self.check_state_exist(self.prevNumToWriteKey)
        self.check_state_exist(self.inMiddleValKey)
        self.check_state_exist(self.inMiddleCountKey)
        self.check_state_exist(self.countCompleteKey)

        self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])
        if numToWrite != self.result_table.ix[self.prevNumToWriteKey, 0]:
            numToWriteKey = "count_" + str(self.countComplete) + "_numToWrite"
            self.result_table.ix[numToWriteKey, 0] = numToWrite
            self.result_table.ix[self.prevNumToWriteKey, 0] = numToWrite
        
        self.sumReward = self.result_table.ix[self.inMiddleValKey, 0]
        self.numRuns = self.result_table.ix[self.inMiddleCountKey, 0]

        if self.numRuns >= numToWrite:
            self.insertEndRun2Table()

    def check_state_exist(self, state):
        if state not in self.result_table.index:
            # append new state to q table
            self.result_table = self.result_table.append(pd.Series([0] * len(self.rewardCol), index=self.result_table.columns, name=state))

    def insertEndRun2Table(self):
            avgReward = self.sumReward / self.numRuns
            countKey = str(self.countComplete)
            self.check_state_exist(countKey)
            self.result_table.ix[countKey, 0] = avgReward

            self.countComplete += 1
            self.result_table.ix[self.countCompleteKey, 0] = self.countComplete

            self.sumReward = 0
            self.numRuns = 0

    def end_run(self, r):
        self.sumReward += r
        self.numRuns += 1

        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()
        
        self.result_table.ix[self.inMiddleValKey, 0] = self.sumReward
        self.result_table.ix[self.inMiddleCountKey, 0] = self.numRuns
        self.result_table.to_pickle(self.tableName + '.gz', 'gzip') 