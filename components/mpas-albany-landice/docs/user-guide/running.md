# Running Cases

This page summarizes how MALI-specific runtime files are generated in E3SM case workflows.

## CIME Integration Path

MALI uses [cime_config/buildnml](../../cime_config/buildnml) to:

1. Read case XML values such as `GLC_GRID`, run type, and coupling flags.
2. Resolve grid-specific input mesh and decomposition metadata.
3. Generate `mali.input_data_list` in the case build area.
4. Run [bld/build-namelist](../../bld/build-namelist) to produce `mali_in`.
5. Copy `mali_in` and streams to the run directory.

## Grid-Dependent Inputs

`buildnml` maps `GLC_GRID` to mesh and related files (mesh, decomposition, optional region mask, optional TF parameters, optional datamode files).

If a grid is unsupported, buildnml exits with an error.

## Restart/Branch/Hybrid Behavior

For `RUN_TYPE` of `branch` or `hybrid`, buildnml expects restart input files and validates their existence before continuing.

## Streams Generation

If a custom streams file exists at `SourceMods/src.mali/streams.landice`, it is used. Otherwise, a default streams file is generated from case metadata.

## Recommended Validation

After case setup/build, verify:

- `mali_in` exists in the run directory.
- `streams.landice` is present and points to expected input files.
- `mali.input_data_list` references files available on your filesystem.

For generic end-to-end case commands, use the E3SM user guide and machine-specific workflow already used by your team.

## Quick-Start Case Recipe

The following is a compact example workflow for a new MALI case in E3SM.
Adjust machine, project, compset, and resolution for your environment.

```bash
# From E3SM root
cd cime/scripts

# Create a case (example values)
./create_newcase \
	--case /path/to/cases/mali_test \
	--compset ITESTMALI \
	--res f19_g16_gl4 \
	--machine <machine> \
	--project <project>

cd /path/to/cases/mali_test

# Optional: tune XML settings before setup
./xmlchange STOP_OPTION=ndays,STOP_N=5
./xmlchange REST_OPTION=ndays,REST_N=5

# Setup and build
./case.setup
./case.build

# Preview/inspect generated namelists
./preview_namelists

# Run
./case.submit
```

After `case.build`, verify MALI-specific outputs in build/run directories:

- `mali_in`
- `streams.landice`
- `mali.input_data_list`

If using branch/hybrid runs, make sure expected restart files exist before buildnml runs.
