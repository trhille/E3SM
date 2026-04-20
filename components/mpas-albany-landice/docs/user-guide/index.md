# User Guide

This guide is intended for users running MALI through E3SM workflows.

## Prerequisites

- Access to an E3SM checkout and input data.
- A supported compiler/MPI environment for your machine.
- Familiarity with standard E3SM case creation and run workflows.

For generic E3SM workflow details, see:

- [E3SM Installation](https://docs.e3sm.org/E3SM/installation/)
- [E3SM User Guide](https://docs.e3sm.org/E3SM/user-guide/)

## Guide Contents

1. [Building MALI](building.md): compiler targets, common flags, and practical build commands.
2. [Namelist Configuration](configuration.md): grouped namelist options and where defaults come from.
3. [Running Cases](running.md): how MALI namelists and streams are generated in CIME runs.

## Notes

- MALI supports multiple build targets in [Makefile](../../Makefile).
- This guide excludes detailed Albany setup; only references to the build flag are provided.
