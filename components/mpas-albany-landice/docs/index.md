# MALI

MPAS-Albany Land Ice (MALI) is the E3SM land-ice component. This documentation provides a quick-start user guide for building and configuring MALI in E3SM.

## Scope

This initial documentation set focuses on:

- Building and compiling MALI.
- Namelist configuration structure and high-impact options.
- Running MALI in E3SM cases.

This phase intentionally excludes detailed Albany setup instructions.

## User Guide

- [Overview](user-guide/index.md)
- [Building MALI](user-guide/building.md)
- [Namelist Configuration](user-guide/configuration.md)
- [Running Cases](user-guide/running.md)

## Source of Truth

These pages describe behavior from:

- [bld/build-namelist](../bld/build-namelist)
- [bld/namelist_files/namelist_definition_mali.xml](../bld/namelist_files/namelist_definition_mali.xml)
- [bld/namelist_files/namelist_defaults_mali.xml](../bld/namelist_files/namelist_defaults_mali.xml)
- [cime_config/buildnml](../cime_config/buildnml)
- [Makefile](../Makefile)
- [src/build_options.mk](../src/build_options.mk)
