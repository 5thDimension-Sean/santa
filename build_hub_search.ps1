$ErrorActionPreference = 'Stop'
$compiler = (Get-Command g++).Source
$jobs = @(
    @('hub_vote.cpp','santa-hub-vote.dll'),
    @('hub_vote.cpp','santa-hub-distance.dll'),
    @('hub_paths.cpp','santa-hub-paths.dll'),
    @('hub_paths.cpp','santa-hub-index.dll','-DCACHE_POOL')
)
foreach ($job in $jobs) {
    $source = Join-Path $PSScriptRoot $job[0]
    $output = Join-Path $env:TEMP $job[1]
    $flags = @('-std=c++17','-O3','-shared','-static')
    if ($job.Count -eq 3) { $flags += $job[2] }
    & $compiler @flags $source -o $output
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed for $source" }
}
& $compiler -std=c++17 -O3 -static (Join-Path $PSScriptRoot 'joint_chain.cpp') -o (Join-Path $env:TEMP 'santa-joint-chain.exe')
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed for joint_chain.cpp' }
