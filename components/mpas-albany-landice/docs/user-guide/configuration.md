# Namelist Configuration

MALI namelist defaults and allowed options are defined by:

- [bld/namelist_files/namelist_definition_mali.xml](../../bld/namelist_files/namelist_definition_mali.xml)
- [bld/namelist_files/namelist_defaults_mali.xml](../../bld/namelist_files/namelist_defaults_mali.xml)
- [bld/build-namelist](../../bld/build-namelist)

## How Namelists Are Built

At case build time, CIME calls [cime_config/buildnml](../../cime_config/buildnml), which invokes [bld/build-namelist](../../bld/build-namelist). The resolved namelist is written to `mali_in` and copied into the run directory.

User overrides are supplied through `user_nl_mali` in the case directory.

## Namelist Groups

MALI groups options as listed in [bld/build-namelist-group-list](../../bld/build-namelist-group-list):

- `velocity_solver`
- `advection`
- `solidearth`
- `calving`
- `thermal_solver`
- `iceshelf_melt`
- `physical_parameters`
- `time_integration`
- `time_management`
- `io`
- `decomposition`
- `debug`
- `subglacial_hydro`
- `am_globalstats`
- `am_regionalstats`

## Group-by-Group Key Parameters

The following parameters are a practical starting set for each group, based on
[bld/build-namelist-section](../../bld/build-namelist-section).

### velocity_solver

- `config_velocity_solver`
- `config_flowParamA_calculation`
- `config_simple_velocity_type`
- `config_nonconvergence_error`

### advection

- `config_thickness_advection`
- `config_tracer_advection`
- `config_horiz_tracer_adv_order`
- `config_advection_coef_3rd_order`
- `config_restore_thickness_after_advection`

### solidearth

- `config_uplift_method`
- `config_slm_coupling_interval`
- `config_MALI_to_SLM_weights_file`
- `config_SLM_to_MALI_weights_file`

### calving

- `config_calving`
- `config_apply_calving_mask`
- `config_calving_thickness`
- `config_calving_timescale`
- `config_remove_icebergs`
- `config_calculate_damage`
- `config_damage_calving_method`
- `config_damage_calving_threshold`

### thermal_solver

- `config_thermal_solver`
- `config_thermal_calculate_bmb`
- `config_temperature_init`
- `config_surface_air_temperature_source`
- `config_basal_heat_flux_source`

### iceshelf_melt

- `config_basal_mass_bal_float`
- `config_front_mass_bal_grounded`
- `config_use_3d_thermal_forcing_for_face_melt`
- `config_uniform_face_melt_rate`
- `config_ocean_data_extrapolation`

### physical_parameters

- `config_ice_density`
- `config_ocean_density`
- `config_sea_level`
- `config_default_flowParamA`
- `config_flowLawExponent`

### time_integration

- `config_dt`
- `config_time_integration`
- `config_rk_order`
- `config_adaptive_timestep`
- `config_adaptive_timestep_CFL_fraction`

### time_management

- `config_do_restart`
- `config_restart_timestamp_name`
- `config_calendar_type`
- `config_start_time`

### io

- `config_stats_interval`
- `config_write_output_on_startup`
- `config_pio_num_iotasks`
- `config_pio_stride`

### decomposition

- `config_num_halos`
- `config_block_decomp_file_prefix`
- `config_number_of_blocks`

### debug

- `config_print_thickness_advection_info`
- `config_print_calving_info`
- `config_print_thermal_info`
- `config_check_tracer_monotonicity`

### subglacial_hydro

- `config_SGH`
- `config_SGH_adaptive_timestep_fraction`
- `config_SGH_till_drainage`
- `config_SGH_basal_melt`
- `config_SGH_chnl_active`

### am_globalstats

- `config_AM_globalStats_enable`
- `config_AM_globalStats_compute_interval`
- `config_AM_globalStats_stream_name`

### am_regionalstats

- `config_AM_regionalStats_enable`
- `config_AM_regionalStats_compute_interval`
- `config_AM_regionalStats_stream_name`

Exact definitions, allowed values, and defaults should always be taken from the XML files above.

## Practical Override Workflow

1. Build a case normally so default `mali_in` is generated.
2. Add or edit entries in `user_nl_mali`.
3. Run preview/build-namelists in your case workflow.
4. Inspect generated `mali_in` in the run/build directories to confirm values.

## Caution

Because namelist defaults can be grid-dependent and mode-dependent, always regenerate namelists after changing grid, run type, or coupling-related XML values.
