
$run=0

DO
{
    $run++
    Taskkill /IM SC2_x64.exe /F
    python .\runSC2Agent.py @args
} While($run -lt 4)