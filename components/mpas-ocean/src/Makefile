.SUFFIXES: .F .c .o

ifeq "$(EXCLUDE_INIT_MODE)" "true"
CPPLOCALFLAGS = -DEXCLUDE_INIT_MODE
$(info "EXCLUDE_INIT_MODE found")
endif

FRAMEWORK_DIR = ../../../mpas-framework/src
OCEAN_SHARED_INCLUDES = -I$(FRAMEWORK_DIR)/framework -I$(FRAMEWORK_DIR)/external/esmf_time_f90 -I$(FRAMEWORK_DIR)/operators
OCEAN_SHARED_INCLUDES += -I$(PWD)/shared -I$(PWD)/analysis_members -I$(PWD)/mode_forward -I$(PWD)/mode_analysis
OCEAN_SHARED_INCLUDES += -I$(PWD)/BGC -I$(PWD)/cvmix/src/shared -I$(PWD)/gotm/build -I$(PWD)/gotm/build/modules
ifneq "$(EXCLUDE_INIT_MODE)" "true"
	OCEAN_SHARED_INCLUDES += -I$(PWD)/mode_init
endif

GOTM_CMAKE_FLAGS = -DGOTM_BUILD_LIBRARIES_ONLY=true -DGOTM_USE_FABM=false -DGOTM_USE_NetCDF=false -DCMAKE_Fortran_COMPILER="$(FC)" -DCMAKE_Fortran_FLAGS="$(FFLAGS)"
ifeq "$(DEBUG)" "true"
	GOTM_CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Debug
endif

all: shared libcvmix analysis_members libBGC libgotm
	(cd mode_forward; $(MAKE) FCINCLUDES="$(FCINCLUDES) $(OCEAN_SHARED_INCLUDES)" all )
	(cd mode_analysis; $(MAKE) FCINCLUDES="$(FCINCLUDES) $(OCEAN_SHARED_INCLUDES)" all )
ifneq "$(EXCLUDE_INIT_MODE)" "true"
	(cd mode_init; $(MAKE) FCINCLUDES="$(FCINCLUDES) $(OCEAN_SHARED_INCLUDES)" all )
endif
	(cd driver; $(MAKE) FCINCLUDES="$(FCINCLUDES) $(OCEAN_SHARED_INCLUDES)" all )
	if [ -e libdycore.a ]; then \
		($(RM) libdycore.a) \
	fi
	ar -ru libdycore.a `find . -type f -name "*.o"`

core_reg:
	$(CPP) $(CPPFLAGS) $(CPPLOCALFLAGS) $(CPPINCLUDES) Registry.xml > Registry_processed.xml

core_input_gen:
	if [ ! -e default_inputs ]; then  mkdir default_inputs; fi
	(cd default_inputs; $(NL_GEN) ../Registry_processed.xml namelist.ocean )
	(cd default_inputs; $(NL_GEN) ../Registry_processed.xml namelist.ocean.forward mode=forward )
	(cd default_inputs; $(NL_GEN) ../Registry_processed.xml namelist.ocean.analysis mode=analysis )
	(cd default_inputs; $(ST_GEN) ../Registry_processed.xml streams.ocean stream_list.ocean. mutable )
	(cd default_inputs; $(ST_GEN) ../Registry_processed.xml streams.ocean.forward stream_list.ocean.forward. mutable mode=forward )
	(cd default_inputs; $(ST_GEN) ../Registry_processed.xml streams.ocean.analysis stream_list.ocean.analysis. mutable mode=analysis )
ifneq "$(EXCLUDE_INIT_MODE)" "true"
	(cd default_inputs; $(NL_GEN) ../Registry_processed.xml namelist.ocean.init mode=init )
	(cd default_inputs; $(ST_GEN) ../Registry_processed.xml streams.ocean.init stream_list.ocean.init. mutable mode=init )
endif

gen_includes:
	$(CPP) $(CPPFLAGS) $(CPPLOCALFLAGS) $(CPPINCLUDES) Registry.xml > Registry_processed.xml
	(if [ ! -d inc ]; then mkdir -p inc; fi) # To generate *.inc files
	(cd inc; $(REG_PARSE) < ../Registry_processed.xml )

post_build:
	if [ ! -e $(ROOT_DIR)/default_inputs ]; then mkdir $(ROOT_DIR)/default_inputs; fi
	cp default_inputs/* $(ROOT_DIR)/default_inputs/.
	( cp $(ROOT_DIR)/default_inputs/namelist.ocean $(ROOT_DIR)/namelist.ocean )
	( cp $(ROOT_DIR)/default_inputs/namelist.ocean.forward $(ROOT_DIR)/namelist.ocean.forward )
	( cp $(ROOT_DIR)/default_inputs/namelist.ocean.analysis $(ROOT_DIR)/namelist.ocean.analysis )
	( cp $(ROOT_DIR)/default_inputs/streams.ocean $(ROOT_DIR)/streams.ocean )
	( cp $(ROOT_DIR)/default_inputs/streams.ocean.forward $(ROOT_DIR)/streams.ocean.forward )
	( cp $(ROOT_DIR)/default_inputs/streams.ocean.analysis $(ROOT_DIR)/streams.ocean.analysis )
ifneq "$(EXCLUDE_INIT_MODE)" "true"
	( cp $(ROOT_DIR)/default_inputs/namelist.ocean.init $(ROOT_DIR)/namelist.ocean.init )
	( cp $(ROOT_DIR)/default_inputs/streams.ocean.init $(ROOT_DIR)/streams.ocean.init )
endif

libcvmix:
	if [ -e cvmix/.git ]; then \
		(cd cvmix/src/shared; make all FC="$(FC)" FCFLAGS="$(FFLAGS)" FINCLUDES="$(FINCLUDES)") \
	else \
		(pwd ; echo "Missing mpas-ocean/src/cvmix/.git, did you forget to 'git submodule update --init --recursive' ?"; exit 1) \
	fi

libBGC:
	if [ -e BGC/.git ]; then \
		(cd BGC; make all FC="$(FC)" FCFLAGS="$(FFLAGS)" FINCLUDES="$(FINCLUDES)") \
	else \
		(echo "Missing mpas-ocean/src/BGC/.git, did you forget to 'git submodule update --init --recursive' ?"; exit 1) \
	fi

libgotm:
	if [ -e gotm/.git ]; then \
		if [ -e gotm/build ]; then \
			(cd gotm/build; make) \
		else \
			(sed -i.bk "s/\#define STDERR write(stderr,\*)/\#define STDERR if (.false.) write(stderr,\*)/" gotm/include/cppdefs.h; mkdir gotm/build; cd gotm/build; cmake ../src $(GOTM_CMAKE_FLAGS); make) \
		fi \
	else \
		(pwd ; echo "Missing core_ocean/gotm/.git, did you forget to 'git submodule update --init --recursive' ?"; exit 1) \
	fi

shared: libcvmix libBGC libgotm
	(cd shared; $(MAKE) FCINCLUDES="$(FCINCLUDES) $(OCEAN_SHARED_INCLUDES)")

analysis_members: libcvmix shared libgotm
	( cd analysis_members; $(MAKE) FCINCLUDES="$(FCINCLUDES) $(OCEAN_SHARED_INCLUDES)" CPPFLAGS="$(CPPFLAGS)" CPPINCLUDES="$(CPPINCLUDES)" all )

clean:
	if [ -e cvmix/.git ]; then \
		(cd cvmix/src/shared; make clean) \
	fi
	if [ -d inc ]; then \
		($(RM) -r inc) \
	fi
	if [ -e BGC/.git ]; then \
		(cd BGC; make clean) \
	fi
	if [ -e gotm/.git ]; then \
		($(RM) -r gotm/build) \
	fi
	(cd mode_forward; $(MAKE) clean)
	(cd mode_analysis; $(MAKE) clean)
	(cd mode_init; $(MAKE) clean)
	(cd driver; $(MAKE) clean)
	(cd analysis_members; $(MAKE) clean)
	(cd shared; $(MAKE) clean)
	($(RM) *.mod libdycore.a Registry_processed.xml)
	$(RM) -r default_inputs
