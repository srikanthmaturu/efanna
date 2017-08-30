cmake_minimum_required(VERSION 3.4)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AppendCompilerFlags)

enable_testing()

project(tf_idf_falconn_index C CXX)

append_cxx_compiler_flags("-msse4.2 -std=c++14 -Wall -DNDEBUG -fopenmp -march=native -DINFO" "GCC" CMAKE_CXX_FLAGS)
append_cxx_compiler_flags("-O3 -ffast-math -funroll-loops" "GCC" CMAKE_CXX_FLAGS)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wall -lboost_timer -lboost_chrono -lboost_system -DINFO")

set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(${CMAKE_HOME_DIRECTORY}
        ${CMAKE_HOME_DIRECTORY}/algorithm
        ${CMAKE_HOME_DIRECTORY}/general
        ${CMAKE_HOME_DIRECTORY}/tf_idf_index
        ${CMAKE_HOME_DIRECTORY}/src/)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

set(BOOST_LIBRARYDIR "/util/opt/boost/1.59/gcc/5.4/lib")
set(BOOST_INCLUDEDIR "/util/opt/boost/1.59/gcc/5.4/include")

find_package(Boost COMPONENTS serialization filesystem timer REQUIRED)

if(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    message("Boost Version: ${Boost_VERSION}")
    message("Boost found")
else()
    message("Boost not found")
endif()

ADD_SUBDIRECTORY(lib)

FILE(STRINGS ${CMAKE_HOME_DIRECTORY}/exp0.config exp0.lines REGEX "^[^#].*")
FOREACH(line ${exp0.lines})
    MESSAGE("line = ${line}")
    LIST(GET line 0 index_name)
    LIST(GET line 1 index_type)
    MESSAGE("Exp 0 ${index_name}; ${index_type};")
    STRING(REGEX REPLACE "tf_idf_efanna_idx<(.*)>" "\\1" t_part ${index_type})
    STRING(REGEX REPLACE "," ";" t_params ${t_part})
    LIST(GET t_params 0 DATA_TYPE)
    LIST(GET t_params 1 NGRAM_LENGTH)
    LIST(GET t_params 2 USE_TDFS)
    LIST(GET t_params 3 USE_IIDF)
    LIST(GET t_params 4 TOTAL_NUMBER_OF_TREES)
    LIST(GET t_params 5 CONQUER_TO_DEPTH)
    LIST(GET t_params 6 ITERATION_NUMBER)
    LIST(GET t_params 7 L)
    LIST(GET t_params 8 CHECK)
    LIST(GET t_params 9 K)
    LIST(GET t_params 10 S)
    LIST(GET t_params 11 NUMBER_OF_TREES_FOR_BUILDING_GRAPH)
    SET(exec ${index_name}_NGL_${NGRAM_LENGTH}_UTD_${USE_TDFS}_UIIDF_${USE_IIDF}_TNT_${TOTAL_NUMBER_OF_TREES}_CTD_${CONQUER_TO_DEPTH}_IN_${ITERATION_NUMBER}_L_${L}_CHECK_${CHECK}_K_${K}_S_${S}_NTBG_${NUMBER_OF_TREES_FOR_BUILDING_GRAPH})
    ADD_EXECUTABLE(${exec} src/index.cpp)
    TARGET_LINK_LIBRARIES(${exec} ${Boost_LIBRARIES})
    SET_PROPERTY(TARGET ${exec} PROPERTY COMPILE_DEFINITIONS
            INDEX_NAME="${index_name}"
            INDEX_TYPE=${index_type}
            DATA_TYPE=${DATA_TYPE}
            NGRAM_LENGTH=${NGRAM_LENGTH}
            USE_TDFS=${USE_TDFS}
            USE_IIDF=${USE_IIDF}
            TOTAL_NUMBER_OF_TREES=${TOTAL_NUMBER_OF_TREES}
            CONQUER_TO_DEPTH=${CONQUER_TO_DEPTH}
            ITERATION_NUMBER=${ITERATION_NUMBER}
            L_CD=${L}
            CHECK=${CHECK}
            K_CD=${K}
            S_CD=${S}
            NUMBER_OF_TREES_FOR_BUILDING_GRAPH=${NUMBER_OF_TREES_FOR_BUILDING_GRAPH}
            )
    message("Target ${exec}")
    message("Index Type ${index_type}")
ENDFOREACH()
