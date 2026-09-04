---
name: repo-map
description: Build a concise architecture map of the current repository.
disable-model-invocation: true
context: fork
agent: Explore
model: haiku
effort: low
---

Create a compact repository map.

Do not read every file.

Use directory listings, grep, config files, entrypoints and imports
to infer architecture.

Ignore:
- .git
- node_modules
- checkpoints
- datasets
- logs
- outputs
- build
- dist

Return:

## Purpose

## Entry points

## Major modules
module -> responsibility

## Important data flows

## Configuration

## Tests

## Important commands

## Key files

Keep the result under 800 words.