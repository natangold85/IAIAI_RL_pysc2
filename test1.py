import os

numDirs = 80
numGenToMove = 10
fromDir = "./TrainArmyAttackGenProgTest2_old/ArmyAttack/armyAttack_A2C_"
origFname = "armyAttack_A2C_result_"
toDir = "./TrainArmyAttackGenProgTest2/army_attack/army_attack_A2C_"
toFName = "results_"

for i in range(numDirs):
    for gen in range(numGenToMove):
        fromDirFName = toDir + str(i) + "/" + origFname + str(gen) + ".gz"
        toDirFName = toDir + str(i) + "/" + toFName + str(gen) + ".gz"
        
        if os.path.isfile(fromDirFName):
            if not os.path.isfile(toDirFName):
                os.rename(fromDirFName, toDirFName)
                print(fromDirFName, "--->", toDirFName)
    
