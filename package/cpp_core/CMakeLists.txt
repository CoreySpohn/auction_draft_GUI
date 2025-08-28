# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.15...3.27)

# Declare the project name
project(cpp_core LANGUAGES CXX)

# Bring in fmt using CMake's FetchContent (no system dependency required)
include(FetchContent)
set(FMT_GIT_TAG 11.2.0)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        ${FMT_GIT_TAG}
)
FetchContent_MakeAvailable(fmt)
# Bring in Eigen using CMake's FetchContent (no system dependency required)
include(FetchContent)
set(EIGEN_GIT_TAG 3.4.0)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        ${EIGEN_GIT_TAG}
)
FetchContent_MakeAvailable(Eigen)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ uv pip install nanobind scikit-build-core[pyproject]
   $ uv pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Python interpreter and development libraries from the active environment
# This is a critical step. The 'Development.Module' component is required by nanobind
find_package(Python 3.12 
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# Find the nanobind package installed via pip
# This command executes a Python script to get the path to nanobind's CMake files
# It is the most robust method for finding a pip-installed nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)

# Now that we have the path, use find_package to import nanobind's CMake functions
find_package(nanobind CONFIG REQUIRED)

# Define our Python extension module
# The first argument, 'simulation_core', is the name of the output file (e.g., simulation_core.so)
# This name MUST match the first argument of the NB_MODULE macro in bindings.cpp
nanobind_add_module(
    cpp_core
    
    # By default nanobind optimizes for size with the (-Os) flag
    # NOMINSIZE is a flag that tells the compiler to optimize for speed (-O3)
    NOMINSIZE
    
    # This targets the stable ABI for Python 3.12+, which
    # reduces the number of binary wheels that are built
    # and does nothing for older Python versions
    STABLE_ABI
    
    # Build libnanobind statically and merge it into the
    # extension (which itself remains a shared library)
    #
    # If your project builds multiple extensions, you can
    # replace this flag by NB_SHARED to conserve space by
    # reusing a shared libnanobind across libraries
    NB_STATIC

    # Reference all source code here
    src/bindings.cpp
    src/agent.cpp
    src/auction.cpp
    src/simulator.cpp
    src/test/hello.cpp
    include/cpp_core/player.hpp
    include/cpp_core/projection.hpp
    include/cpp_core/agent.hpp
    include/cpp_core/market.hpp
    include/cpp_core/draft_state.hpp
    include/cpp_core/auction.hpp
    include/cpp_core/simulator.hpp
    # Add other.cpp files here as they are created, e.g.:
    # src/Agent.cpp
    # src/Simulation.cpp
)

# Link fmt (header-only) to the extension module to avoid extra linkage
target_link_libraries(cpp_core PRIVATE fmt::fmt-header-only)
target_link_libraries(cpp_core PRIVATE Eigen3::Eigen)

# Install directive for scikit-build-core
install(
    TARGETS cpp_core
    LIBRARY DESTINATION .
)

# Make public headers available as #include "cpp_core/..."
target_include_directories(cpp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)