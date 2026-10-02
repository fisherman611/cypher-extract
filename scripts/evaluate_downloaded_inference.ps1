[CmdletBinding()]
param(
    [int[]]$Seeds = @(10, 42, 50, 100, 1234),
    [ValidateSet("cypherbench", "mind_the_query", "neo4j_text2cypher")]
    [string[]]$Datasets = @("cypherbench", "mind_the_query", "neo4j_text2cypher"),
    [string[]]$Metrics = @("execution_accuracy", "psjs", "executable"),
    [string]$Python = "python",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Evaluator = Join-Path $PSScriptRoot "evaluate_cypher_all.ps1"
$InferenceBase = Join-Path $RepositoryRoot "results/inference"
$EvaluationBase = Join-Path $RepositoryRoot "results/evaluation"

if (-not (Test-Path -LiteralPath $Evaluator -PathType Leaf)) {
    throw "Evaluation script does not exist: $Evaluator"
}

if (-not (Test-Path -LiteralPath $InferenceBase -PathType Container)) {
    throw "Downloaded inference directory does not exist: $InferenceBase"
}

# CypherBench and Mind-the-Query use the local Neo4j instance. The underlying
# evaluator loads .env itself; this check only fails early with a useful message.
$hasPassword = -not [string]::IsNullOrWhiteSpace($env:NEO4J_PASSWORD)
$envFile = Join-Path $RepositoryRoot ".env"
if (-not $hasPassword -and (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    $passwordLine = Get-Content -LiteralPath $envFile |
        Where-Object { $_ -match '^\s*NEO4J_PASSWORD\s*=\s*(.+?)\s*$' } |
        Select-Object -First 1
    if ($passwordLine) {
        $password = ($passwordLine -split '=', 2)[1].Trim().Trim('"').Trim("'")
        $hasPassword = $password -and $password -ne "change-me"
    }
}
if (-not $hasPassword) {
    throw "Set NEO4J_PASSWORD in $envFile or in the current environment before running evaluation."
}

$allDatasets = @("cypherbench", "mind_the_query", "neo4j_text2cypher")
$jobs = @(
    [PSCustomObject]@{
        Name = "lora"
        Methods = @("sft", "teacher_lora")
        Datasets = $allDatasets
    },
    [PSCustomObject]@{
        Name = "lora"
        Methods = @(
            "amid", "csd", "distillm_adaptive_sfkl", "distillm_adaptive_srkl", "fdd_sfkl",
            "fdd_srkl", "fkl", "hpd", "rkl", "sfkl", "srkl"
        )
        Datasets = @("cypherbench")
    },
    [PSCustomObject]@{
        Name = "full_finetune"
        Methods = @("sft")
        Datasets = $allDatasets
    },
    [PSCustomObject]@{
        Name = "lora_normalized"
        Methods = @("teacher_lora")
        Datasets = $allDatasets
    },
    [PSCustomObject]@{
        Name = "full_finetune_normalized"
        Methods = @("sft")
        Datasets = @("cypherbench")
    }
)

foreach ($job in $jobs) {
    $selectedDatasets = @($job.Datasets | Where-Object { $_ -in $Datasets })
    if ($selectedDatasets.Count -eq 0) {
        continue
    }

    $inferenceRoot = Join-Path $InferenceBase "$($job.Name)/qwen3"
    $evaluationRoot = Join-Path $EvaluationBase "$($job.Name)/qwen3"
    if (-not (Test-Path -LiteralPath $inferenceRoot -PathType Container)) {
        throw "Inference setting does not exist: $inferenceRoot"
    }

    Write-Host "[$($job.Name)] starting evaluation" -ForegroundColor Cyan
    $arguments = @{
        InferenceRoot = $inferenceRoot
        EvaluationRoot = $evaluationRoot
        Seeds = $Seeds
        Methods = $job.Methods
        Datasets = $selectedDatasets
        Metrics = $Metrics
        Python = $Python
        SkipExisting = (-not $Force)
    }
    $global:LASTEXITCODE = 0
    & $Evaluator @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for setting '$($job.Name)' (exit $LASTEXITCODE)."
    }
    Write-Host "[$($job.Name)] complete" -ForegroundColor Green
}

Write-Host "All downloaded inference outputs were evaluated successfully." -ForegroundColor Green
