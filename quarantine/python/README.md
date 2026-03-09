# Quarantined Python Surface

Start in `AGENTS.md`.

This directory holds standalone Python scripts that are not part of the active stable
worktree surface.

## Why These Files Live Here
- they are not owned by an official manifest or adapter
- they are not documented as bounded repo-skill tools
- they are not pure library-only modules

## Rules
- preserve the original relative path under `quarantine/python/`
- do not point official manifests, adapters, or start-path docs at files in this tree
- if a script becomes active again, re-onboard it through a manifest, adapter, or skill, or
  refactor it into a library-only module before moving it back

Git history remains the fallback forensic surface for older experiment behavior.
