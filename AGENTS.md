# Project Overview

This repository contains analysis and post-processing code for the
April 2026 coronal-hole simulation using SIP-IFVM.

The project uses three main directories:

## Source data

E:/Research/Data/SIP-IFVM

This directory contains SIP-IFVM simulation data, including HDF5 files,
merged spherical grids, and other source data.

IMPORTANT:
- Treat this directory as READ-ONLY unless explicitly instructed otherwise.
- Never overwrite or delete source simulation data.

## Code

E:/Research/Program/SIP-IFVM

This is the main code repository.

Codex may:
- inspect source code;
- modify Python/Fortran/C++ code when requested;
- create new scripts;
- run tests;
- run analysis scripts;
- refactor code when requested.

## Research output

E:/Research/Work/Coronal_hole_by_SIP

All newly generated research outputs should normally be written here,
including:

- figures;
- NPZ files;
- processed HDF5 files;
- derived datasets;
- logs;
- animations;
- analysis results.

Existing files in this directory may also be used as input for later
analysis.

Do not overwrite important existing research results unless explicitly
requested.

# SIP-IFVM coordinate system

Primary spherical variables:

r
theta
phi

Magnetic field:

Br
Btheta
Bphi

Velocity:

Vr
Vtheta
Vphi

theta is colatitude unless otherwise stated by the existing code.

# Physical units

Density:
usph(1) * (Rhos / 1.672E-27) / 1E6
unit: cm^-3

Velocity:
usph(2:4) * Vs / 1000
unit: km/s

Magnetic field:
usph(6:8) * Bs * 1E4
unit: Gauss

# Coding rules

When modifying existing code:

1. Preserve existing physical definitions unless explicitly asked to change them.
2. Preserve comments unless modification is necessary.
3. Do not silently change coordinate conventions.
4. Do not silently change units.
5. Do not silently change output filenames or formats.
6. Prefer using Br, Btheta, Bphi directly when available.
7. Avoid hard-coded paths when practical.
8. Keep paths configurable near the top of scripts.

# Execution policy

When asked to implement or modify a script:

1. Inspect relevant existing code first.
2. Make the requested modification.
3. Run syntax checks.
4. Run the script when practical.
5. Inspect terminal errors.
6. Fix errors caused by the modification.
7. Verify generated files exist.
8. Report where outputs were written.

Do not stop after merely writing code if the user explicitly asks for
results to be generated.

# Scientific validation

For numerical analysis:

- check array dimensions;
- check NaN/Inf;
- check coordinate transformations;
- check physical units;
- preserve numerical precision where possible;
- compare with previous output when appropriate.

A script completing without error does not by itself prove that the
scientific result is correct.