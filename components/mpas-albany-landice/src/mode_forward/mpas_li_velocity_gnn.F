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
   use mpi

   use li_setup
   use li_mask
   use li_constants

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

   ! Global edge_index for the GNN (0-based, bidirectional)
   integer(kind=8), dimension(:,:), allocatable, target, save :: global_edge_index

   ! MPI gather/scatter bookkeeping
   integer, save :: nCellsGlobal          ! total number of cells across all ranks
   integer, save :: nLocalOwned           ! nCellsSolve on this rank
   integer, save :: nDirectedEdgesGlobal  ! number of directed edges in global_edge_index
   integer, allocatable, save :: recvcounts(:)  ! number of owned cells per rank
   integer, allocatable, save :: displs(:)      ! displacements for gatherv/scatterv

   ! Device type for FTorch
   integer, save :: ftorch_device_type

   ! Number of input features
   integer, parameter :: nFeatures = 11

   logical, save :: gnn_initialized = .false.

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
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: meshPool
      integer, pointer :: nCellsSolve, nEdgesSolve
      integer, dimension(:,:), pointer :: cellsOnEdge
      integer, dimension(:), pointer :: indexToCellID, indexToEdgeID
      integer :: myRank, nProcs, iEdge, idx
      integer :: nEdgesLocal
      ! Temporary arrays for gathering edge connectivity
      integer, allocatable :: local_edges(:,:)  ! (2, nEdgesSolve) with global cell IDs
      integer, allocatable :: edgecounts(:), edgedispls(:)
      integer, allocatable :: all_edges_flat(:)
      integer :: nEdgesAll, i

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
      ! Collect local edges as global cell ID pairs
      allocate(local_edges(2, nEdgesLocal))
      do iEdge = 1, nEdgesLocal
         local_edges(1, iEdge) = indexToCellID(cellsOnEdge(1, iEdge))
         local_edges(2, iEdge) = indexToCellID(cellsOnEdge(2, iEdge))
      end do

      ! Gather edge counts from all ranks
      allocate(edgecounts(nProcs))
      allocate(edgedispls(nProcs))
      call MPI_Allgather(nEdgesLocal * 2, 1, MPI_INTEGER, edgecounts, 1, MPI_INTEGER, &
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

      call MPI_Gatherv(local_edges, nEdgesLocal * 2, MPI_INTEGER, &
                       all_edges_flat, edgecounts, edgedispls, MPI_INTEGER, &
                       0, domain % dminfo % comm, err)

      ! On rank 0, build bidirectional edge_index with 0-based indexing
      if (myRank == 0) then
         nDirectedEdgesGlobal = nEdgesAll * 2
         allocate(global_edge_index(2, nDirectedEdgesGlobal))
         idx = 0
         do i = 1, nEdgesAll
            ! Forward direction
            idx = idx + 1
            global_edge_index(1, idx) = all_edges_flat((i-1)*2 + 1) - 1  ! 0-based
            global_edge_index(2, idx) = all_edges_flat((i-1)*2 + 2) - 1  ! 0-based
            ! Reverse direction
            idx = idx + 1
            global_edge_index(1, idx) = all_edges_flat((i-1)*2 + 2) - 1  ! 0-based
            global_edge_index(2, idx) = all_edges_flat((i-1)*2 + 1) - 1  ! 0-based
         end do
      else
         nDirectedEdgesGlobal = 0
         allocate(global_edge_index(2, 1))  ! placeholder
      endif

      deallocate(local_edges, edgecounts, edgedispls, all_edges_flat)

      ! Broadcast nDirectedEdgesGlobal to all ranks (needed for consistency)
      call MPI_Bcast(nDirectedEdgesGlobal, 1, MPI_INTEGER, 0, domain % dminfo % comm, err)

      ! --- Load the TorchScript model on rank 0 ---
      if (myRank == 0) then
         call torch_model_load(gnn_model, trim(config_gnn_model_path), ftorch_device_type)
         call mpas_log_write('GNN velocity model loaded from: ' // trim(config_gnn_model_path))
      endif

      gnn_initialized = .true.
      call mpas_log_write('GNN velocity solver initialized.')
      call mpas_log_write('  Global cells: $i, Directed edges: $i', &
           intArgs=(/nCellsGlobal, nDirectedEdgesGlobal/))

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
!>  normalVelocity fields.
!
!-----------------------------------------------------------------------

   subroutine li_velocity_gnn_solve(domain, err)

      type (domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      ! local variables
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: meshPool, geometryPool, thermalPool, velocityPool
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

      ! Global arrays on rank 0
      real (kind=RKIND), allocatable, target :: global_features(:,:)   ! (nCellsGlobal, nFeatures)
      real (kind=RKIND), allocatable, target :: global_velocity(:,:)   ! (nCellsGlobal, 2)
      integer(kind=8), allocatable, target :: global_batch(:)          ! (nCellsGlobal) all zeros

#ifdef USE_FTORCH
      type(torch_tensor), dimension(3) :: in_tensors
      type(torch_tensor), dimension(1) :: out_tensors
#endif

      integer :: iCell, iEdge, k, myRank, cell1, cell2
      integer :: nVertInterfacesLocal
      real (kind=RKIND) :: xVelEdge, yVelEdge
      real (kind=RKIND) :: thicknessAboveFloatation, depthAvgTemp
      integer, pointer :: nVertLevels

      err = 0

#ifdef USE_FTORCH
      if (.not. gnn_initialized) then
         call mpas_log_write('GNN velocity solver not initialized.', MPAS_LOG_ERR)
         err = 1
         return
      endif

      myRank = domain % dminfo % my_proc_id

      ! --- Get pools from block ---
      block => domain % blocklist
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
      call mpas_pool_get_subpool(block % structs, 'geometry', geometryPool)
      call mpas_pool_get_subpool(block % structs, 'thermal', thermalPool)
      call mpas_pool_get_subpool(block % structs, 'velocity', velocityPool)

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
      end do

      ! --- MPI Gather features to rank 0 ---
      if (myRank == 0) then
         allocate(global_features(nCellsGlobal, nFeatures))
         allocate(global_velocity(nCellsGlobal, 2))
         allocate(global_batch(nCellsGlobal))
         global_batch(:) = 0  ! single graph: all nodes belong to batch 0
      else
         allocate(global_features(1, 1))  ! placeholder
         allocate(global_velocity(1, 1))  ! placeholder
         allocate(global_batch(1))        ! placeholder
      endif

      ! Gather each feature column separately to handle Fortran column-major layout
      call gather_features_to_rank0(local_features, global_features, &
                                    nLocalOwned, nCellsGlobal, nFeatures, &
                                    recvcounts, displs, domain % dminfo % comm, myRank)

      ! --- Run inference on rank 0 ---
      if (myRank == 0) then
         call torch_tensor_from_array(in_tensors(1), global_features, ftorch_device_type)
         call torch_tensor_from_array(in_tensors(2), global_edge_index, ftorch_device_type)
         call torch_tensor_from_array(in_tensors(3), global_batch, ftorch_device_type)
         call torch_tensor_from_array(out_tensors(1), global_velocity, torch_kCPU)

         call torch_model_forward(gnn_model, in_tensors, out_tensors)

         call torch_delete(in_tensors)
         call torch_delete(out_tensors)
      endif

      ! --- Scatter predicted velocities back to all ranks ---
      allocate(local_velocity(nLocalOwned, 2))
      call scatter_velocity_from_rank0(global_velocity, local_velocity, &
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

      ! --- Compute normalVelocity from cell-centered velocities ---
      do iEdge = 1, nEdges
         cell1 = cellsOnEdge(1, iEdge)
         cell2 = cellsOnEdge(2, iEdge)
         do k = 1, nVertInterfacesLocal
            xVelEdge = 0.5_RKIND * (uReconstructX(k, cell1) + uReconstructX(k, cell2))
            yVelEdge = 0.5_RKIND * (uReconstructY(k, cell1) + uReconstructY(k, cell2))
            normalVelocity(k, iEdge) = xVelEdge * edgeNormalVectors(1, iEdge) + &
                                       yVelEdge * edgeNormalVectors(2, iEdge)
         end do
      end do

      ! Clean up
      deallocate(local_features, local_velocity, global_features, global_velocity, global_batch)

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
         if (allocated(global_edge_index)) deallocate(global_edge_index)
         if (allocated(recvcounts)) deallocate(recvcounts)
         if (allocated(displs)) deallocate(displs)
         gnn_initialized = .false.
      endif
#endif

   end subroutine li_velocity_gnn_finalize


!***********************************************************************
! Private helper routines
!***********************************************************************

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


end module li_velocity_gnn
