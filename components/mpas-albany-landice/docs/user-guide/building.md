# Building MALI

This page documents practical build usage from the component [Makefile](../../Makefile) and [src/build_options.mk](../../src/build_options.mk).

## Build Location

Run builds from the component root:

```bash
cd components/mpas-albany-landice
```

The Makefile auto-detects the core from the path and builds `landice_model` for this component.

## Common Build Commands

Example command used on many Cray/GNU environments:

```bash
make gnu-cray DEBUG=true
```

Optimized build:

```bash
make gnu-cray
```

Clean build products:

```bash
make clean
```

Show build usage and options:

```bash
make errmsg
```

## High-Impact Build Options

From the Makefile help and build logic:

- `DEBUG=true`: enables debug flags and floating-point traps per target toolchain.
- `OPENMP=true`: builds with OpenMP flags.
- `USE_PAPI=true`: enables PAPI timers.
- `USE_LAPACK=true`: links LAPACK/BLAS.
- `USE_PIO2=true`: links PIO2 instead of PIO1.
- `AUTOCLEAN=true`: forces infrastructure clean before rebuilding the core.

## External Dycore Flags

From [src/build_options.mk](../../src/build_options.mk):

- `LIFEV=true` enables external dycore interfaces for L1L2 and FO.
- `ALBANY=true` enables external FO interface compile-time flags.
- `PHG=true` requires `LIFEV=true`.
- `LIFEV=true` and `ALBANY=true` together are disallowed.

This user guide does not include Albany environment/setup instructions.

## Environment Variables

The Makefile expects relevant library paths in the environment when enabled, including `NETCDF`, `PNETCDF`, and `PIO` (and others depending on chosen options).

## Typical Debug Build Checklist

1. Start from a clean state: `make clean`.
2. Build with debug enabled: `make gnu-cray DEBUG=true`.
3. Confirm executable creation: `landice_model` in component root.
4. If needed, rerun with `OPENMP=true` or profiling flags.
