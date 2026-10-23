! Copyright (c) 2013-2018,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  li_velocity_gnn
!
!> \brief MPAS land-ice GNN velocity emulator
!> \author Trevor Hillebrand
!> \date   June 2026
!> \details
!>  This module couples a PyTorch Geometric GNN velocity emulator
!>  to MALI via FTorch. The GNN predicts depth-averaged x/y velocities
!>  from cell-based features and graph connectivity.
!>  Data is gathered to rank 0 for global inference and scattered back.
!
!-----------------------------------------------------------------------

module li_velocity_gnn

#ifdef USE_FTORCH
   use ftorch, only: torch_model, torch_tensor, torch_kCPU, torch_kCUDA, &
                     torch_tensor_from_array, torch_model_load, &
                     torch_model_forward, torch_delete
#endif

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_log
   use mpas_timer
   use mpi

   use li_setup
   use li_mask
   use li_constants
   use ieee_arithmetic
   use ieee_exceptions
#ifdef USE_FTORCH
   use iso_c_binding, only: c_int, c_int64_t, c_double, c_char, c_null_char
#endif

   implicit none
   private

   public :: li_velocity_gnn_init, &
             li_velocity_gnn_solve, &
             li_velocity_gnn_finalize

   !--------------------------------------------------------------------
   ! Private module variables (persistent state)
   !--------------------------------------------------------------------
#ifdef USE_FTORCH
   type(torch_model), save :: gnn_model
#endif

   ! Global undirected edge pairs (0-based cell indices, built once in init)
   integer, dimension(:,:), allocatable, save :: global_edge_pairs
   integer, save :: nUndirectedEdgesGlobal  ! number of undirected edges

   ! MPI gather/scatter bookkeeping
   integer, save :: nCellsGlobal          ! total number of cells across all ranks
   integer, save :: nLocalOwned           ! nCellsSolve on this rank
   integer, allocatable, save :: recvcounts(:)  ! number of owned cells per rank
   integer, allocatable, save :: displs(:)      ! displacements for gatherv/scatterv

   ! Device type for FTorch
   integer, save :: ftorch_device_type

   ! Number of input features
   integer, parameter :: nFeatures = 12

   logical, save :: gnn_initialized = .false.

   !--------------------------------------------------------------------
   ! Random-forest error check (optional; see config_gnn_rf_check).  Flags
   ! GNN solves that are likely to have high velocity error so that
   ! li_velocity (mpas_li_velocity.F) can fall back to the external ('FO')
   ! dycore for that solve instead of trusting the GNN prediction.
   !--------------------------------------------------------------------
   integer, parameter :: RF_SLOT_X = 0  ! forest slot for x-velocity error check
   integer, parameter :: RF_SLOT_Y = 1  ! forest slot for y-velocity error check
   ! Matching the slot args' bind(C) kind avoids relying on automatic
   ! integer-kind conversion for VALUE dummy arguments.
#ifdef USE_FTORCH
   integer(c_int), parameter :: RF_SLOT_X_C = int(RF_SLOT_X, c_int)
   integer(c_int), parameter :: RF_SLOT_Y_C = int(RF_SLOT_Y, c_int)
#endif

   logical, save :: rfCheckEnabled = .false.
   integer, save :: rfThreshold = 0          ! config_gnn_rf_threshold
   real (kind=RKIND), save :: rfSampleFraction = 0.0_RKIND  ! config_gnn_rf_sample_fraction
   integer, save :: rfBaseSeed = 42          ! config_gnn_rf_seed
   ! gnnSolveCount and gnnFallbackCount are summary-logging tallies only (reset to 0
   ! on restart, like any plain module variable) -- NOT the RF check's seed source.
   ! The seed instead comes from the restart-persistent pool field
   ! 'gnnRFCheckSolveCount' (see li_velocity_gnn_solve), so that the RF check's
   ! random subsampling is reproducible across a restart even though this run tally
   ! is not.
   integer, save :: gnnSolveCount = 0        ! number of GNN solves attempted this run (since last restart)
   integer, save :: gnnFallbackCount = 0     ! number of those solves that fell back to the external dycore

#ifdef USE_FTORCH
   !--------------------------------------------------------------------
   ! C bindings to the random-forest predictor (mpas_li_rf_predict.cpp)
   !--------------------------------------------------------------------
   interface

      function li_rf_load(slot, path) result(ierr) bind(C, name="li_rf_load")
         import :: c_int, c_char
         integer(c_int), value :: slot
         character(kind=c_char), dimension(*), intent(in) :: path
         integer(c_int) :: ierr
      end function li_rf_load

      function li_rf_count_ones(slot, x, nRows, nFeat, nSample, seed, nOnes, nTies) result(ierr) &
            bind(C, name="li_rf_count_ones")
         import :: c_int, c_int64_t, c_double
         integer(c_int), value :: slot
         real(c_double), dimension(*), intent(in) :: x
         integer(c_int), value :: nRows, nFeat, nSample
         integer(c_int64_t), value :: seed
         integer(c_int), intent(out) :: nOnes
         integer(c_int), intent(out) :: nTies
         integer(c_int) :: ierr
      end function li_rf_count_ones

      subroutine li_rf_free(slot) bind(C, name="li_rf_free")
         import :: c_int
         integer(c_int), value :: slot
      end subroutine li_rf_free

      subroutine li_rf_last_error(buf, buflen) bind(C, name="li_rf_last_error")
         import :: c_char, c_int
         character(kind=c_char), dimension(*), intent(out) :: buf
         integer(c_int), value :: buflen
      end subroutine li_rf_last_error

   end interface
#endif

contains

!***********************************************************************
!
!  routine li_velocity_gnn_init
!
!> \brief   Initialize GNN velocity solver
!> \author  Trevor Hillebrand
!> \date    June 2026
!> \details
!>  Load TorchScript model, build global edge_index, set up MPI
!>  gather/scatter infrastructure.
!
!-----------------------------------------------------------------------

   subroutine li_velocity_gnn_init(domain, err)

      type (domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      ! local variables
      character (len=StrKIND), pointer :: config_gnn_model_path
      character (len=StrKIND), pointer :: config_gnn_device_type
      logical, pointer :: config_gnn_rf_check
      character (len=StrKIND), pointer :: config_gnn_rf_weights_x
      character (len=StrKIND), pointer :: config_gnn_rf_weights_y
      real (kind=RKIND), pointer :: config_gnn_rf_sample_fraction
      integer, pointer :: config_gnn_rf_threshold
      integer, pointer :: config_gnn_rf_seed
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: meshPool
      integer, pointer :: nCellsSolve, nEdgesSolve
      integer, dimension(:,:), pointer :: cellsOnEdge
      integer, dimension(:), pointer :: indexToCellID, indexToEdgeID
      integer :: myRank, nProcs, iEdge, idx
      integer :: nEdgesLocal, nValidEdges
      ! Temporary arrays for gathering edge connectivity
      integer, allocatable :: local_edges(:,:)  ! (2, nValidEdges) with global cell IDs
      integer, allocatable :: edgecounts(:), edgedispls(:)
      integer, allocatable :: all_edges_flat(:)
      integer :: nEdgesAll, i
#ifdef USE_FTORCH
      integer(c_int) :: rfLoadErr
      integer :: rfInitErr, mpiErr
#endif

      err = 0

#ifdef USE_FTORCH
      call mpas_pool_get_config(liConfigs, 'config_gnn_model_path', config_gnn_model_path)
      call mpas_pool_get_config(liConfigs, 'config_gnn_device_type', config_gnn_device_type)

      ! Set device type
      if (trim(config_gnn_device_type) == 'cuda') then
         ftorch_device_type = torch_kCUDA
      else
         ftorch_device_type = torch_kCPU
      endif

      myRank = domain % dminfo % my_proc_id
      nProcs = domain % dminfo % nProcs

      ! --- Determine local owned cells and set up gather/scatter ---
      block => domain % blocklist
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
      call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
      call mpas_pool_get_dimension(meshPool, 'nEdgesSolve', nEdgesSolve)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)

      nLocalOwned = nCellsSolve
      nEdgesLocal = nEdgesSolve

      ! Gather nCellsSolve from each rank to build displacements
      allocate(recvcounts(nProcs))
      allocate(displs(nProcs))
      call MPI_Allgather(nLocalOwned, 1, MPI_INTEGER, recvcounts, 1, MPI_INTEGER, &
                         domain % dminfo % comm, err)
      nCellsGlobal = sum(recvcounts)
      displs(1) = 0
      do i = 2, nProcs
         displs(i) = displs(i-1) + recvcounts(i-1)
      end do

      ! --- Build global edge_index ---
      ! Collect local edges as global cell ID pairs, skipping boundary
      ! edges where cellsOnEdge contains 0 (garbage cells)
      nValidEdges = 0
      do iEdge = 1, nEdgesLocal
         if (cellsOnEdge(1, iEdge) > 0 .and. cellsOnEdge(2, iEdge) > 0) then
            nValidEdges = nValidEdges + 1
         endif
      end do

      allocate(local_edges(2, nValidEdges))
      idx = 0
      do iEdge = 1, nEdgesLocal
         if (cellsOnEdge(1, iEdge) > 0 .and. cellsOnEdge(2, iEdge) > 0) then
            idx = idx + 1
            local_edges(1, idx) = indexToCellID(cellsOnEdge(1, iEdge))
            local_edges(2, idx) = indexToCellID(cellsOnEdge(2, iEdge))
         endif
      end do

      ! Gather edge counts from all ranks
      allocate(edgecounts(nProcs))
      allocate(edgedispls(nProcs))
      call MPI_Allgather(nValidEdges * 2, 1, MPI_INTEGER, edgecounts, 1, MPI_INTEGER, &
                         domain % dminfo % comm, err)
      nEdgesAll = sum(edgecounts) / 2  ! total undirected edges
      edgedispls(1) = 0
      do i = 2, nProcs
         edgedispls(i) = edgedispls(i-1) + edgecounts(i-1)
      end do

      ! Gather all edges to rank 0
      if (myRank == 0) then
         allocate(all_edges_flat(sum(edgecounts)))
      else
         allocate(all_edges_flat(1))  ! placeholder
      endif

      call MPI_Gatherv(local_edges, nValidEdges * 2, MPI_INTEGER, &
                       all_edges_flat, edgecounts, edgedispls, MPI_INTEGER, &
                       0, domain % dminfo % comm, err)

      ! On rank 0, store undirected edge pairs with 0-based indexing
      if (myRank == 0) then
         nUndirectedEdgesGlobal = nEdgesAll
         allocate(global_edge_pairs(2, nUndirectedEdgesGlobal))
         do i = 1, nUndirectedEdgesGlobal
            global_edge_pairs(1, i) = all_edges_flat((i-1)*2 + 1) - 1  ! 0-based
            global_edge_pairs(2, i) = all_edges_flat((i-1)*2 + 2) - 1  ! 0-based
         end do
      else
         nUndirectedEdgesGlobal = 0
         allocate(global_edge_pairs(2, 1))  ! placeholder
      endif

      deallocate(local_edges, edgecounts, edgedispls, all_edges_flat)

      ! Broadcast nUndirectedEdgesGlobal to all ranks (needed for consistency)
      call MPI_Bcast(nUndirectedEdgesGlobal, 1, MPI_INTEGER, 0, domain % dminfo % comm, err)

      ! --- Load the TorchScript model on rank 0 ---
      if (myRank == 0) then
         call torch_model_load(gnn_model, trim(config_gnn_model_path), ftorch_device_type)
         call mpas_log_write('GNN velocity model loaded from: ' // trim(config_gnn_model_path))
      endif

      ! --- Optionally load the random-forest error-check forests on rank 0 ---
      call mpas_pool_get_config(liConfigs, 'config_gnn_rf_check', config_gnn_rf_check)
      rfCheckEnabled = config_gnn_rf_check
      if (rfCheckEnabled) then
         call mpas_pool_get_config(liConfigs, 'config_gnn_rf_weights_x', config_gnn_rf_weights_x)
         call mpas_pool_get_config(liConfigs, 'config_gnn_rf_weights_y', config_gnn_rf_weights_y)
         call mpas_pool_get_config(liConfigs, 'config_gnn_rf_sample_fraction', config_gnn_rf_sample_fraction)
         call mpas_pool_get_config(liConfigs, 'config_gnn_rf_threshold', config_gnn_rf_threshold)
         call mpas_pool_get_config(liConfigs, 'config_gnn_rf_seed', config_gnn_rf_seed)
         rfSampleFraction = config_gnn_rf_sample_fraction
         rfThreshold = config_gnn_rf_threshold
         rfBaseSeed = config_gnn_rf_seed

         rfInitErr = 0
         if (myRank == 0) then
            rfLoadErr = li_rf_load(RF_SLOT_X_C, trim(config_gnn_rf_weights_x) // c_null_char)
            if (rfLoadErr /= 0) then
               call log_rf_error('Failed to load GNN RF x-velocity error-check forest from ' // &
                  trim(config_gnn_rf_weights_x) // ': ')
               rfInitErr = 1
            else
               call mpas_log_write('GNN RF x-velocity error-check forest loaded from: ' // &
                  trim(config_gnn_rf_weights_x))
            endif

            rfLoadErr = li_rf_load(RF_SLOT_Y_C, trim(config_gnn_rf_weights_y) // c_null_char)
            if (rfLoadErr /= 0) then
               call log_rf_error('Failed to load GNN RF y-velocity error-check forest from ' // &
                  trim(config_gnn_rf_weights_y) // ': ')
               rfInitErr = 1
            else
               call mpas_log_write('GNN RF y-velocity error-check forest loaded from: ' // &
                  trim(config_gnn_rf_weights_y))
            endif
         endif

         ! Broadcast load success/failure so all ranks abort together rather
         ! than only rank 0 detecting the error.
         call MPI_Bcast(rfInitErr, 1, MPI_INTEGER, 0, domain % dminfo % comm, mpiErr)
         if (rfInitErr /= 0) then
            call mpas_log_write('GNN random-forest error check (config_gnn_rf_check) failed to initialize.', MPAS_LOG_CRIT)
            err = 1
            return
         endif

#ifndef USE_EXTERNAL_FIRSTORDER
         call mpas_log_write('config_gnn_rf_check = .true. requires compiling with the external FO dycore ' // &
            '(USE_EXTERNAL_FIRSTORDER), since a flagged solve falls back to it.', MPAS_LOG_CRIT)
         err = 1
         return
#endif

         call mpas_log_write('GNN random-forest error check enabled.  sample_fraction=$r threshold=$i seed=$i', &
              realArgs=(/rfSampleFraction/), intArgs=(/rfThreshold, rfBaseSeed/))
      endif

      gnn_initialized = .true.
      call mpas_log_write('GNN velocity solver initialized.')
      call mpas_log_write('  Global cells: $i, Undirected edges: $i', &
           intArgs=(/nCellsGlobal, nUndirectedEdgesGlobal/))
      if (myRank == 0) then
         call mpas_log_write('nCellsGlobal: $i', intArgs=(/nCellsGlobal/))
         call mpas_log_write('edge_pairs min: $i', intArgs=(/minval(global_edge_pairs)/))
         call mpas_log_write('edge_pairs max: $i', intArgs=(/maxval(global_edge_pairs)/))
      endif

#else
      call mpas_log_write('GNN velocity solver requires compilation with USE_FTORCH.', MPAS_LOG_CRIT)
      err = 1
#endif

   end subroutine li_velocity_gnn_init


!***********************************************************************
!
!  routine li_velocity_gnn_solve
!
!> \brief   Solve velocity using GNN emulator
!> \author  Trevor Hillebrand
!> \date    June 2026
!> \details
!>  Gathers input features to rank 0, runs GNN inference, scatters
!>  predicted velocities back, and fills uReconstructX/Y and
!>  normalVelocity fields.  If config_gnn_rf_check is enabled, a
!>  random-forest error check is run on the GNN input features (on a random
!>  subsample of ice cells) before the GNN inference.  If either forest
!>  flags more sampled cells than config_gnn_rf_threshold, GNN inference is
!>  skipped and useFallback is set so the caller (li_velocity_solve in
!>  mpas_li_velocity.F) falls back to the external dycore for this solve.
!
!-----------------------------------------------------------------------

   subroutine li_velocity_gnn_solve(domain, useFallback, err)

      type (domain_type), intent(inout) :: domain
      logical, intent(out) :: useFallback !< Output: .true. if the random-forest
      !< error check flagged this solve; the caller should fall back to the
      !< external dycore rather than using GNN-predicted velocities.
      integer, intent(out) :: err

      ! local variables
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: meshPool, geometryPool, thermalPool, velocityPool
      integer, pointer :: gnnRFCheckSolveCount
      integer, pointer :: nCellsSolve, nEdges, nVertInterfaces
      integer, dimension(:), pointer :: cellMask, indexToCellID
      integer, dimension(:,:), pointer :: cellsOnEdge
      real (kind=RKIND), dimension(:), pointer :: areaCell, thickness, bedTopography
      real (kind=RKIND), dimension(:), pointer :: xCell, yCell
      real (kind=RKIND), dimension(:), pointer :: muFriction, stiffnessFactor
      real (kind=RKIND), dimension(:,:), pointer :: temperature
      real (kind=RKIND), dimension(:), pointer :: basalTemperature
      real (kind=RKIND), dimension(:), pointer :: layerThicknessFractions
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocity
      real (kind=RKIND), dimension(:,:), pointer :: uReconstructX, uReconstructY
      real (kind=RKIND), dimension(:,:), pointer :: edgeNormalVectors
      real (kind=RKIND), pointer :: config_sea_level, config_ice_density, config_ocean_density

      ! Local feature assembly
      real (kind=RKIND), allocatable :: local_features(:,:)  ! (nLocalOwned, nFeatures)
      real (kind=RKIND), allocatable :: local_velocity(:,:)  ! (nLocalOwned, 2)

      ! Global arrays on rank 0 (indexed by global cell ID)
      real (kind=RKIND), allocatable :: global_features(:,:)   ! (nCellsGlobal, nFeatures)
      real (kind=RKIND), allocatable :: global_velocity(:,:)   ! (nCellsGlobal, 2)
      real (kind=RKIND), allocatable :: gathered_features(:,:) ! (nCellsGlobal, nFeatures) in gather order
      real (kind=RKIND), allocatable :: gathered_velocity(:,:) ! (nCellsGlobal, 2) in gather order
      integer, allocatable :: global_cellIDs(:)                ! (nCellsGlobal) maps gathered pos -> global cell ID

      ! Ice-only subgraph arrays (rank 0 only)
      integer, allocatable :: ice_map(:)                                ! maps global cell ID to compact ice index
      integer :: nIceCells, nIceEdgesUndirected, nIceEdgesDirected
      real (kind=4), allocatable, target :: ice_features(:,:)           ! (nIceCells, nFeatures) single precision
      real (kind=4), allocatable, target :: ice_velocity(:,:)           ! (nIceCells, 2) single precision
      integer(kind=8), allocatable, target :: ice_edge_index(:,:)       ! (2, nIceEdgesDirected)
      integer(kind=8), allocatable, target :: ice_batch(:)              ! (nIceCells)

#ifdef USE_FTORCH
      type(torch_tensor), dimension(3) :: in_tensors
      type(torch_tensor), dimension(1) :: out_tensors
#endif

      integer :: iCell, iEdge, i, idx, k, myRank, cell1, cell2, c1, c2
      integer :: nVertInterfacesLocal
      real (kind=RKIND) :: xVelEdge, yVelEdge
      real (kind=RKIND) :: thicknessAboveFloatation, depthAvgTemp
      integer, pointer :: nVertLevels
      logical :: fpe_invalid, fpe_divzero, fpe_overflow  ! saved FPE trap state

      ! Random-forest error check (rank 0 only; see li_velocity_gnn_init).
      ! li_rf_count_ones() draws its own random subsample internally (given
      ! nSample and a seed), so the full ice-cell feature matrix is passed
      ! through, just promoted from single to double precision; a Fortran
      ! (nIceCells, nFeatures) array is already stored column-major, which
      ! is the layout li_rf_count_ones expects.
#ifdef USE_FTORCH
      real (c_double), allocatable :: ice_features_dp(:,:)  ! (nIceCells, nFeatures)
      integer(c_int) :: rfOnesX, rfOnesY, rfTiesX, rfTiesY
      integer(c_int) :: rfErrX, rfErrY
      integer(c_int64_t) :: rfSeed
#endif
      integer :: nSample
      integer :: fallbackInt, mpiErr2

      err = 0
      useFallback = .false.

#ifdef USE_FTORCH
      if (.not. gnn_initialized) then
         call mpas_log_write('GNN velocity solver not initialized.', MPAS_LOG_ERR)
         err = 1
         return
      endif

      ! Computed on rank 0 only below, then broadcast; initialize here so
      ! every rank has a defined value going into that broadcast.
      fallbackInt = 0

      myRank = domain % dminfo % my_proc_id

      ! --- Get pools from block ---
      block => domain % blocklist
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
      call mpas_pool_get_subpool(block % structs, 'geometry', geometryPool)
      call mpas_pool_get_subpool(block % structs, 'thermal', thermalPool)
      call mpas_pool_get_subpool(block % structs, 'velocity', velocityPool)

      ! gnnRFCheckSolveCount is a restart-persistent pool field (unlike a plain
      ! module variable) so the RF error check's per-solve seed below stays
      ! consistent with a continuous run regardless of where this run restarted.
      call mpas_pool_get_array(velocityPool, 'gnnRFCheckSolveCount', gnnRFCheckSolveCount)
      gnnRFCheckSolveCount = gnnRFCheckSolveCount + 1
      gnnSolveCount = gnnSolveCount + 1  ! this-run-only tally, for the summary log in li_velocity_gnn_finalize

      call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
      call mpas_pool_get_dimension(meshPool, 'nVertInterfaces', nVertInterfaces)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)

      nVertInterfacesLocal = nVertInterfaces

      ! Get mesh arrays
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'xCell', xCell)
      call mpas_pool_get_array(meshPool, 'yCell', yCell)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'edgeNormalVectors', edgeNormalVectors)
      call mpas_pool_get_array(meshPool, 'layerThicknessFractions', layerThicknessFractions)
      call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)

      ! Get geometry arrays
      call mpas_pool_get_array(geometryPool, 'thickness', thickness)
      call mpas_pool_get_array(geometryPool, 'bedTopography', bedTopography)
      call mpas_pool_get_array(geometryPool, 'cellMask', cellMask)

      ! Get thermal arrays
      call mpas_pool_get_array(thermalPool, 'temperature', temperature)
      call mpas_pool_get_array(thermalPool, 'basalTemperature', basalTemperature)

      ! Get velocity arrays
      call mpas_pool_get_array(velocityPool, 'muFriction', muFriction)
      call mpas_pool_get_array(velocityPool, 'stiffnessFactor', stiffnessFactor)
      call mpas_pool_get_array(velocityPool, 'normalVelocity', normalVelocity)
      call mpas_pool_get_array(velocityPool, 'uReconstructX', uReconstructX)
      call mpas_pool_get_array(velocityPool, 'uReconstructY', uReconstructY)

      ! Get configs
      call mpas_pool_get_config(liConfigs, 'config_sea_level', config_sea_level)
      call mpas_pool_get_config(liConfigs, 'config_ice_density', config_ice_density)
      call mpas_pool_get_config(liConfigs, 'config_ocean_density', config_ocean_density)

      ! --- Assemble local node features ---
      allocate(local_features(nLocalOwned, nFeatures))
      do iCell = 1, nLocalOwned
         ! Feature 1: xCell (km)
         local_features(iCell, 1) = xCell(iCell) / 1000.0_RKIND

         ! Feature 2: yCell (km)
         local_features(iCell, 2) = yCell(iCell) / 1000.0_RKIND

         ! Feature 3: areaCell (km^2)
         local_features(iCell, 3) = areaCell(iCell) / 1.0e6_RKIND

         ! Feature 4: thickness (km)
         local_features(iCell, 4) = thickness(iCell) / 1000.0_RKIND

         ! Feature 5: bedTopography (km)
         local_features(iCell, 5) = bedTopography(iCell) / 1000.0_RKIND

         ! Feature 6: log10(muFriction)
         if (muFriction(iCell) > 0.0_RKIND) then
            local_features(iCell, 6) = log10(muFriction(iCell))
         else
            local_features(iCell, 6) = 0.0_RKIND
         endif

         ! Feature 7: basalTemperature (°C)
         local_features(iCell, 7) = basalTemperature(iCell) - kelvin_to_celsius

         ! Feature 8: depth-averaged temperature (°C)
         depthAvgTemp = 0.0_RKIND
         do k = 1, nVertLevels
            depthAvgTemp = depthAvgTemp + temperature(k, iCell) * layerThicknessFractions(k)
         end do
         local_features(iCell, 8) = depthAvgTemp - kelvin_to_celsius

         ! Feature 9: stiffnessFactor
         local_features(iCell, 9) = stiffnessFactor(iCell)

         ! Feature 10: thicknessAboveFloatation (km)
         thicknessAboveFloatation = thickness(iCell) - &
            max(0.0_RKIND, (config_sea_level - bedTopography(iCell)) * &
                config_ocean_density / config_ice_density)
         local_features(iCell, 10) = thicknessAboveFloatation / 1000.0_RKIND

         ! Feature 11: floatMask (1.0 if floating, 0.0 otherwise)
         if (li_mask_is_floating_ice(cellMask(iCell))) then
            local_features(iCell, 11) = 1.0_RKIND
         else
            local_features(iCell, 11) = 0.0_RKIND
         endif

         ! Feature 12: margin mask (1.0 if ice margin, 0.0 otherwise)
         if (li_mask_is_margin(cellMask(iCell))) then
            local_features(iCell, 12) = 1.0_RKIND
         else
            local_features(iCell, 12) = 0.0_RKIND
         endif
      end do

      ! --- MPI Gather features and cell IDs to rank 0 ---
      if (myRank == 0) then
         allocate(gathered_features(nCellsGlobal, nFeatures))
         allocate(global_features(nCellsGlobal, nFeatures))
         allocate(global_velocity(nCellsGlobal, 2))
         allocate(gathered_velocity(nCellsGlobal, 2))
         allocate(global_cellIDs(nCellsGlobal))
         global_velocity(:,:) = 0.0_RKIND  ! zero velocity for non-ice cells
      else
         allocate(gathered_features(1, 1))  ! placeholder
         allocate(global_features(1, 1))  ! placeholder
         allocate(global_velocity(1, 1))  ! placeholder
         allocate(gathered_velocity(1, 1))  ! placeholder
         allocate(global_cellIDs(1))  ! placeholder
      endif

      ! Gather cell IDs so we can reorder from gathered position to global cell ID order
      call MPI_Gatherv(indexToCellID, nLocalOwned, MPI_INTEGER, &
                       global_cellIDs, recvcounts, displs, MPI_INTEGER, &
                       0, domain % dminfo % comm, err)

      ! Gather each feature column separately to handle Fortran column-major layout
      call gather_features_to_rank0(local_features, gathered_features, &
                                    nLocalOwned, nCellsGlobal, nFeatures, &
                                    recvcounts, displs, domain % dminfo % comm, myRank)

      ! Reorder features from gathered order to global cell ID order
      ! so that features and edge indices use the same index space
      if (myRank == 0) then
         do i = 1, nCellsGlobal
            global_features(global_cellIDs(i), :) = gathered_features(i, :)
         end do
      endif

      ! --- Filter to ice-only cells on rank 0 ---
      if (myRank == 0) then
         ! Identify ice cells (thickness > 0, feature 4)
         allocate(ice_map(nCellsGlobal))
         nIceCells = 0
         do iCell = 1, nCellsGlobal
            if (global_features(iCell, 4) > 0.0_RKIND) then  ! thickness in km
               nIceCells = nIceCells + 1
               ice_map(iCell) = nIceCells - 1  ! 0-based compact index
            else
               ice_map(iCell) = -1
            endif
         end do

         ! Build ice-only features
         allocate(ice_features(nIceCells, nFeatures))
         idx = 0
         do iCell = 1, nCellsGlobal
            if (ice_map(iCell) >= 0) then
               idx = idx + 1
               ice_features(idx, :) = global_features(iCell, :)
            endif
         end do

         ! Debug: print first ice cell features from global (double) and ice (single)
         call mpas_log_write('First global_features (double): x=$r y=$r thick=$r', &
              realArgs=(/global_features(1, 1), global_features(1, 2), global_features(1, 4)/))
         call mpas_log_write('First ice_features (single): x=$r y=$r thick=$r', &
              realArgs=(/dble(ice_features(1, 1)), dble(ice_features(1, 2)), dble(ice_features(1, 4))/))
         call mpas_log_write('nIceCells=$i, nIceEdgesDirected will be computed next', &
              intArgs=(/nIceCells/))
      endif

      ! --- Random-forest error check (optional) ---
      ! Flags this solve as likely high-error before spending time on GNN
      ! inference. Uses the full ice-cell feature matrix (promoted to
      ! double precision); li_rf_count_ones draws its own deterministic
      ! random subsample internally from a per-solve seed, operating in
      ! global-cell-ID order so the decision is independent of the
      ! domain decomposition (number of MPI ranks/blocks).
      ! The timer start/stop below must run on every rank (not just rank 0,
      ! where the actual work happens) because mpas_timer_write() is a
      ! collective call at the end of the run -- every rank must build an
      ! identical timer tree, or its MPI_Allreduce-based reductions
      ! desynchronize across ranks. rfCheckEnabled is set identically on
      ! every rank in li_velocity_gnn_init, so this guard is already symmetric.
      if (rfCheckEnabled) call mpas_timer_start("gnn rf check")
      if (myRank == 0 .and. rfCheckEnabled) then
         nSample = max(1, nint(rfSampleFraction * real(nIceCells, RKIND)))
         rfSeed = int(rfBaseSeed, c_int64_t) + int(gnnRFCheckSolveCount, c_int64_t)

         allocate(ice_features_dp(nIceCells, nFeatures))
         ice_features_dp = real(ice_features, c_double)

         rfErrX = li_rf_count_ones(RF_SLOT_X_C, ice_features_dp, nIceCells, nFeatures, nSample, &
                                    rfSeed, rfOnesX, rfTiesX)
         rfErrY = li_rf_count_ones(RF_SLOT_Y_C, ice_features_dp, nIceCells, nFeatures, nSample, &
                                    rfSeed, rfOnesY, rfTiesY)
         deallocate(ice_features_dp)

         if (rfErrX /= 0 .or. rfErrY /= 0) then
            call log_rf_error('GNN RF error check failed: ')
            err = 1
         else
            call mpas_log_write('GNN RF error check: nSample=$i x: ones=$i ties=$i, y: ones=$i ties=$i, threshold=$i', &
                 intArgs=(/nSample, int(rfOnesX), int(rfTiesX), int(rfOnesY), int(rfTiesY), rfThreshold/))
            if (int(rfOnesX) > rfThreshold .or. int(rfOnesY) > rfThreshold) then
               fallbackInt = 1
            endif
         endif
      endif
      if (rfCheckEnabled) call mpas_timer_stop("gnn rf check")

      ! --- GNN inference on rank 0 ---
      ! If the random-forest error check flagged this solve, skip GNN
      ! inference entirely; the caller (li_velocity_solve in
      ! mpas_li_velocity.F) will fall back to the external dycore
      ! instead of using global_velocity (left at its initialized value
      ! of zero below).
      ! fallbackInt is 0 on every rank at this point except possibly rank 0
      ! (only rank 0 can set it to 1, above, and only rank 0 has reached this
      ! point with any other value), so "fallbackInt == 0" evaluates .true.
      ! symmetrically on every other rank, keeping this timer's start/stop
      ! calls symmetric across ranks for the same reason as above.
      if (fallbackInt == 0) call mpas_timer_start("gnn inference")
      if (myRank == 0) then
         if (fallbackInt == 0) then

         ! Build ice-only bidirectional edge_index
         ! First count edges where both endpoints have ice
         nIceEdgesUndirected = 0
         do i = 1, nUndirectedEdgesGlobal
            c1 = global_edge_pairs(1, i) + 1  ! 1-based for ice_map lookup
            c2 = global_edge_pairs(2, i) + 1
            if (c1 >= 1 .and. c1 <= nCellsGlobal .and. &
                c2 >= 1 .and. c2 <= nCellsGlobal) then
               if (ice_map(c1) >= 0 .and. ice_map(c2) >= 0) then
                  nIceEdgesUndirected = nIceEdgesUndirected + 1
               endif
            endif
         end do

         nIceEdgesDirected = nIceEdgesUndirected * 2
         allocate(ice_edge_index(2, nIceEdgesDirected))
         idx = 0
         do i = 1, nUndirectedEdgesGlobal
            c1 = global_edge_pairs(1, i) + 1
            c2 = global_edge_pairs(2, i) + 1
            if (c1 >= 1 .and. c1 <= nCellsGlobal .and. &
                c2 >= 1 .and. c2 <= nCellsGlobal) then
               if (ice_map(c1) >= 0 .and. ice_map(c2) >= 0) then
                  idx = idx + 1
                  ice_edge_index(1, idx) = ice_map(c1)
                  ice_edge_index(2, idx) = ice_map(c2)
                  idx = idx + 1
                  ice_edge_index(1, idx) = ice_map(c2)
                  ice_edge_index(2, idx) = ice_map(c1)
               endif
            endif
         end do

         ! Build batch tensor
         allocate(ice_batch(nIceCells))
         ice_batch(:) = 0

         ! Build tensors and run inference
         allocate(ice_velocity(nIceCells, 2))

         ! Disable FPE traps around LibTorch call (LibTorch triggers them internally)
         call ieee_get_halting_mode(ieee_invalid, fpe_invalid)
         call ieee_get_halting_mode(ieee_divide_by_zero, fpe_divzero)
         call ieee_get_halting_mode(ieee_overflow, fpe_overflow)
         call ieee_set_halting_mode(ieee_invalid, .false.)
         call ieee_set_halting_mode(ieee_divide_by_zero, .false.)
         call ieee_set_halting_mode(ieee_overflow, .false.)

         call torch_tensor_from_array(in_tensors(1), ice_features, [1, 2], ftorch_device_type)
         call torch_tensor_from_array(in_tensors(2), ice_edge_index, [1, 2], ftorch_device_type)
         call torch_tensor_from_array(in_tensors(3), ice_batch, [1], ftorch_device_type)
         call torch_tensor_from_array(out_tensors(1), ice_velocity, [1, 2], torch_kCPU)

         call torch_model_forward(gnn_model, in_tensors, out_tensors)

         call torch_delete(in_tensors)
         call torch_delete(out_tensors)

         ! Restore FPE traps
         call ieee_set_halting_mode(ieee_invalid, fpe_invalid)
         call ieee_set_halting_mode(ieee_divide_by_zero, fpe_divzero)
         call ieee_set_halting_mode(ieee_overflow, fpe_overflow)

         ! Map ice velocities back to global array
         do iCell = 1, nCellsGlobal
            if (ice_map(iCell) >= 0) then
               global_velocity(iCell, 1) = ice_velocity(ice_map(iCell) + 1, 1)
               global_velocity(iCell, 2) = ice_velocity(ice_map(iCell) + 1, 2)
            endif
         end do

         call mpas_log_write('GNN inference: $i ice cells, $i directed ice edges', &
              intArgs=(/nIceCells, nIceEdgesDirected/))
         call mpas_log_write('GNN velocity stats (km/yr): vx min=$r max=$r, vy min=$r max=$r', &
              realArgs=(/dble(minval(ice_velocity(:,1))), dble(maxval(ice_velocity(:,1))), &
                         dble(minval(ice_velocity(:,2))), dble(maxval(ice_velocity(:,2)))/))
         call mpas_log_write('GNN velocity cell 1: vx=$r vy=$r (km/yr)', &
              realArgs=(/dble(ice_velocity(1,1)), dble(ice_velocity(1,2))/))

         ! Dump GNN inputs/outputs as text for debugging (first solve call only)
         call dump_gnn_debug(ice_features, ice_edge_index, ice_velocity, &
                             nIceCells, nFeatures, nIceEdgesDirected)

         deallocate(ice_edge_index, ice_velocity, ice_batch)

         else
            call mpas_log_write('GNN random-forest error check flagged this solve; ' // &
               'falling back to the external dycore instead of using the GNN prediction.', MPAS_LOG_WARN)
         endif

         deallocate(ice_map, ice_features)
      endif
      if (fallbackInt == 0) call mpas_timer_stop("gnn inference")

      ! Broadcast the fallback decision (computed on rank 0 only) to all ranks
      call MPI_Bcast(fallbackInt, 1, MPI_INTEGER, 0, domain % dminfo % comm, mpiErr2)
      useFallback = (fallbackInt == 1)
      if (useFallback) gnnFallbackCount = gnnFallbackCount + 1

      ! If this solve was flagged, skip the scatter/fill/halo-exchange below:
      ! the caller (li_velocity_solve in mpas_li_velocity.F) will overwrite
      ! uReconstructX/Y and normalVelocity with an external-dycore solve
      ! instead of using these (never-computed) GNN velocities.
      if (useFallback) then
         deallocate(local_features, global_features, gathered_features)
         deallocate(global_velocity, gathered_velocity, global_cellIDs)
         return
      endif

      ! Reorder velocity from global cell ID order back to gathered order for scattering
      if (myRank == 0) then
         do i = 1, nCellsGlobal
            gathered_velocity(i, :) = global_velocity(global_cellIDs(i), :)
         end do
      endif

      ! --- Scatter predicted velocities back to all ranks ---
      allocate(local_velocity(nLocalOwned, 2))
      call scatter_velocity_from_rank0(gathered_velocity, local_velocity, &
                                       nLocalOwned, nCellsGlobal, &
                                       recvcounts, displs, domain % dminfo % comm, myRank)

      ! --- Fill uReconstructX/Y with uniform vertical profile ---
      ! GNN predicts in km/yr; convert to m/s
      do iCell = 1, nLocalOwned
         do k = 1, nVertInterfacesLocal
            uReconstructX(k, iCell) = local_velocity(iCell, 1) * 1000.0_RKIND / scyr
            uReconstructY(k, iCell) = local_velocity(iCell, 2) * 1000.0_RKIND / scyr
         end do
      end do

      ! Halo exchange uReconstructX/Y so that halo cells have current values
      ! before computing normalVelocity on owned edges with halo-cell neighbors
      call mpas_dmpar_field_halo_exch(domain, 'uReconstructX')
      call mpas_dmpar_field_halo_exch(domain, 'uReconstructY')

      ! --- Compute normalVelocity from cell-centered velocities ---
      ! At margin edges (ice cell <-> non-ice cell), use one-sided velocity
      ! from the ice cell rather than averaging with zero. This allows
      ! advection to transport ice into empty cells, analogous to how Albany
      ! provides velocity on its boundary edges.
      do iEdge = 1, nEdges
         cell1 = cellsOnEdge(1, iEdge)
         cell2 = cellsOnEdge(2, iEdge)
         do k = 1, nVertInterfacesLocal
            if (thickness(cell1) > 0.0_RKIND .and. thickness(cell2) <= 0.0_RKIND) then
               ! Margin edge: only cell1 has ice, use its velocity
               xVelEdge = uReconstructX(k, cell1)
               yVelEdge = uReconstructY(k, cell1)
            else if (thickness(cell2) > 0.0_RKIND .and. thickness(cell1) <= 0.0_RKIND) then
               ! Margin edge: only cell2 has ice, use its velocity
               xVelEdge = uReconstructX(k, cell2)
               yVelEdge = uReconstructY(k, cell2)
            else
               ! Interior edge or both cells empty: average
               xVelEdge = 0.5_RKIND * (uReconstructX(k, cell1) + uReconstructX(k, cell2))
               yVelEdge = 0.5_RKIND * (uReconstructY(k, cell1) + uReconstructY(k, cell2))
            endif
            normalVelocity(k, iEdge) = xVelEdge * edgeNormalVectors(1, iEdge) + &
                                       yVelEdge * edgeNormalVectors(2, iEdge)
         end do
      end do

      ! Clean up
      deallocate(local_features, local_velocity, global_features, global_velocity)
      deallocate(gathered_features, gathered_velocity, global_cellIDs)

#else
      call mpas_log_write('GNN velocity solver requires compilation with USE_FTORCH.', MPAS_LOG_CRIT)
      err = 1
#endif

   end subroutine li_velocity_gnn_solve


!***********************************************************************
!
!  routine li_velocity_gnn_finalize
!
!> \brief   Finalize GNN velocity solver
!> \author  Trevor Hillebrand
!> \date    June 2026
!> \details
!>  Clean up FTorch model and deallocate persistent arrays.
!
!-----------------------------------------------------------------------

   subroutine li_velocity_gnn_finalize(err)

      integer, intent(out) :: err

      err = 0

#ifdef USE_FTORCH
      if (gnn_initialized) then
         call torch_delete(gnn_model)
         if (allocated(global_edge_pairs)) deallocate(global_edge_pairs)
         if (allocated(recvcounts)) deallocate(recvcounts)
         if (allocated(displs)) deallocate(displs)
         if (rfCheckEnabled) then
            call li_rf_free(RF_SLOT_X_C)
            call li_rf_free(RF_SLOT_Y_C)
            call mpas_log_write('GNN random-forest error check: $i of $i solves fell back to the external dycore.', &
                 intArgs=(/gnnFallbackCount, gnnSolveCount/))
         endif
         gnn_initialized = .false.
      endif
#endif

   end subroutine li_velocity_gnn_finalize


!***********************************************************************
! Private helper routines
!***********************************************************************

#ifdef USE_FTORCH
!-----------------------------------------------------------------------
!
!  routine log_rf_error
!
!> \brief   Log a message plus the last error from the RF predictor library
!> \details Fetches the message set by mpas_li_rf_predict.cpp's most recent
!>  failure and writes prefix // message as an MPAS_LOG_ERR.
!
!-----------------------------------------------------------------------

   subroutine log_rf_error(prefix)

      character (len=*), intent(in) :: prefix

      character(kind=c_char), dimension(256) :: errBuf
      character (len=256) :: errMsg
      integer :: i

      call li_rf_last_error(errBuf, size(errBuf))
      errMsg = ' '
      do i = 1, size(errBuf)
         if (errBuf(i) == c_null_char) exit
         errMsg(i:i) = errBuf(i)
      end do

      call mpas_log_write(prefix // trim(errMsg), MPAS_LOG_ERR)

   end subroutine log_rf_error
#endif


!-----------------------------------------------------------------------
!
!  routine gather_features_to_rank0
!
!> \brief   Gather node features from all ranks to rank 0
!
!-----------------------------------------------------------------------

   subroutine gather_features_to_rank0(local_features, global_features, &
                                       nLocal, nGlobal, nFeat, &
                                       counts, disps, comm, myRank)

      real (kind=RKIND), intent(in) :: local_features(:,:)
      real (kind=RKIND), intent(inout) :: global_features(:,:)
      integer, intent(in) :: nLocal, nGlobal, nFeat
      integer, intent(in) :: counts(:), disps(:)
      integer, intent(in) :: comm, myRank

      real (kind=RKIND), allocatable :: local_col(:), global_col(:)
      integer :: iFeat, ierr

      allocate(local_col(nLocal))
      if (myRank == 0) then
         allocate(global_col(nGlobal))
      else
         allocate(global_col(1))
      endif

      do iFeat = 1, nFeat
         local_col(1:nLocal) = local_features(1:nLocal, iFeat)
         call MPI_Gatherv(local_col, nLocal, MPI_DOUBLE_PRECISION, &
                          global_col, counts, disps, MPI_DOUBLE_PRECISION, &
                          0, comm, ierr)
         if (myRank == 0) then
            global_features(1:nGlobal, iFeat) = global_col(1:nGlobal)
         endif
      end do

      deallocate(local_col, global_col)

   end subroutine gather_features_to_rank0


!-----------------------------------------------------------------------
!
!  routine scatter_velocity_from_rank0
!
!> \brief   Scatter predicted velocities from rank 0 to all ranks
!
!-----------------------------------------------------------------------

   subroutine scatter_velocity_from_rank0(global_velocity, local_velocity, &
                                          nLocal, nGlobal, &
                                          counts, disps, comm, myRank)

      real (kind=RKIND), intent(in) :: global_velocity(:,:)
      real (kind=RKIND), intent(out) :: local_velocity(:,:)
      integer, intent(in) :: nLocal, nGlobal
      integer, intent(in) :: counts(:), disps(:)
      integer, intent(in) :: comm, myRank

      real (kind=RKIND), allocatable :: local_col(:), global_col(:)
      integer :: iComp, ierr

      allocate(local_col(nLocal))
      if (myRank == 0) then
         allocate(global_col(nGlobal))
      else
         allocate(global_col(1))
      endif

      ! Scatter x-velocity (column 1) and y-velocity (column 2)
      do iComp = 1, 2
         if (myRank == 0) then
            global_col(1:nGlobal) = global_velocity(1:nGlobal, iComp)
         endif
         call MPI_Scatterv(global_col, counts, disps, MPI_DOUBLE_PRECISION, &
                           local_col, nLocal, MPI_DOUBLE_PRECISION, &
                           0, comm, ierr)
         local_velocity(1:nLocal, iComp) = local_col(1:nLocal)
      end do

      deallocate(local_col, global_col)

   end subroutine scatter_velocity_from_rank0


!-----------------------------------------------------------------------
!
!  routine dump_gnn_debug
!
!> \brief   Write GNN inputs/outputs to binary files for debugging
!> \details Only writes once (first call). Files can be read in Python:
!>    features = np.fromfile('gnn_debug_features.bin', dtype=np.float32).reshape(nIceCells, nFeatures)
!>    edge_index = np.fromfile('gnn_debug_edge_index.bin', dtype=np.int64).reshape(2, nEdges)
!>    velocity = np.fromfile('gnn_debug_velocity.bin', dtype=np.float32).reshape(nIceCells, 2)
!
!-----------------------------------------------------------------------

   subroutine dump_gnn_debug(features, edge_index, velocity, nCells, nFeat, nEdges)

      real (kind=4), intent(in) :: features(:,:)
      integer(kind=8), intent(in) :: edge_index(:,:)
      real (kind=4), intent(in) :: velocity(:,:)
      integer, intent(in) :: nCells, nFeat, nEdges

      logical, save :: dumped = .false.
      integer :: iu, ios
      integer :: i

      if (dumped) return
      dumped = .true.

      call mpas_log_write('Dumping GNN debug text files: nCells=$i, nFeat=$i, nEdges=$i', &
           intArgs=(/nCells, nFeat, nEdges/))

      ! Write all features as text (formatted I/O known to work)
      open(newunit=iu, file='gnn_debug_features.txt', status='replace', iostat=ios)
      write(iu, '(A,I8,A,I3)') '# nCells=', nCells, ' nFeat=', nFeat
      do i = 1, nCells
         write(iu, '(12(ES15.7,1X))') features(i, 1:nFeat)
      end do
      close(iu)

      ! Write all edge pairs as text
      open(newunit=iu, file='gnn_debug_edges.txt', status='replace', iostat=ios)
      write(iu, '(A,I10)') '# nEdges=', nEdges
      do i = 1, nEdges
         write(iu, '(2(I10,1X))') edge_index(1, i), edge_index(2, i)
      end do
      close(iu)

      ! Write all velocities as text
      open(newunit=iu, file='gnn_debug_velocity.txt', status='replace', iostat=ios)
      write(iu, '(A,I8)') '# nCells=', nCells
      do i = 1, nCells
         write(iu, '(2(ES15.7,1X))') velocity(i, 1), velocity(i, 2)
      end do
      close(iu)

      call mpas_log_write('GNN debug text files written.')

   end subroutine dump_gnn_debug


end module li_velocity_gnn
