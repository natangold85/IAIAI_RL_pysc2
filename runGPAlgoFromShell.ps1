
$run=0
$NumGenerations=100
$PopulationSize=20
DO
{
    $run++

    python .\runSC2Agent.py --do=gpTrain --runDir=TrainArmyAttackGenProgTest --trainAgent=army_attack --device=cpu --populationSize=$PopulationSize
    $popIdx=0
    DO
    {
        echo "`n`n`nstart testing pop idx" $popIdx "`n`n`n`n"
        python .\runSC2Agent.py --do=gpTest --map=ArmyAttack5x5 --runDir=TrainArmyAttackGenProgTest --testAgent=army_attack --plot=False --device=cpu --numEpisodes=200 --populationSize=$popIdx
        Taskkill /IM SC2_x64.exe /F
        $popIdx++
    } While($popIdx -lt $PopulationSize)

} While($run -lt $NumGenerations)