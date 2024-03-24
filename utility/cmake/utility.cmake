#重新定义当前目标的源文件的__FILE__宏
function(redefine_file targetname)
    #获取当前源文件的编译参数
    get_property(defs SOURCE "${targetname}" PROPERTY COMPILE_DEFINITIONS)
    #获取当前文件的绝对路径
    get_filename_component(filepath "${targetname}" ABSOLUTE)
    #将绝对路径中的项目路径替换成空,得到源文件相对于项目路径的相对路径
    string(REPLACE ../.. "" relpath ${filepath})
    #将我们要加的编译参数(__FILENAME__定义)添加到原来的编译参数里面
    list(APPEND defs "__FILENAME__=\"${relpath}\"")
    #重新设置源文件的编译参数
    set_property(SOURCE "${targetname}" PROPERTY COMPILE_DEFINITIONS ${defs})
endfunction(redefine_file)

function(redefine_file_base targetname)
    #获取当前源文件的编译参数
    get_property(defs SOURCE "${targetname}" PROPERTY COMPILE_DEFINITIONS)
    #获取当前文件的绝对路径
    get_filename_component(filepath "${targetname}" NAME)
    message(${filepath})
    #将我们要加的编译参数(__FILENAME__定义)添加到原来的编译参数里面
    list(APPEND defs "__FILENAME__=\"${filepath}\"")
    #重新设置源文件的编译参数
    set_property(SOURCE "${targetname}" PROPERTY COMPILE_DEFINITIONS ${defs})
endfunction(redefine_file_base)

find_package(Git QUIET)     # 查找Git，QUIET静默方式不报错
if (GIT_FOUND)
    execute_process(# 执行一个子进程
            COMMAND ${GIT_EXECUTABLE} log -1 --format=%H # 命令
            OUTPUT_VARIABLE GIT_VERSION        # 输出字符串存入变量
            OUTPUT_STRIP_TRAILING_WHITESPACE    # 删除字符串尾的换行符
            ERROR_QUIET                         # 对执行错误静默
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} # 执行路径
    )
    add_definitions(-DGIT_VERSION=${GIT_VERSION})

    execute_process(
            COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
            OUTPUT_VARIABLE GIT_SHORT_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if (GIT_SHORT_VERSION STREQUAL "")
        string(TIMESTAMP GIT_SHORT_VERSION "%H%M%S")
    endif ()
    add_definitions(-DGIT_SHORT_VERSION=${GIT_SHORT_VERSION})

    execute_process(
            COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%an
            OUTPUT_VARIABLE GIT_AUTHOR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if (GIT_AUTHOR STREQUAL "")
        set(GIT_AUTHOR "lucky.liu")
    endif ()
    add_definitions(-DGIT_AUTHOR=${GIT_AUTHOR})

    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list --all --count
            OUTPUT_VARIABLE REVISION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if (REVISION STREQUAL "")
        string(TIMESTAMP REVISION "%Y%m%d")
    endif ()
    math(EXPR REVISION "${REVISION}-${REVISION_PREFIX}")
    add_definitions(-DREVISION=${REVISION})

    string(TIMESTAMP COMPILE_TIME "%Y-%m-%dT%H:%M:%S")
    add_definitions(-DCOMPILE_TIME=${COMPILE_TIME})
    #set(VERSION_MAJOR ${major})    # 一级版本号
    #set(VERSION_MINOR ${minor})    # 二级版本号


    #configure_file(
    #        "config.h.in"
    #        "../config.h"
    #)
endif()

if (NOT DEFINED UTILITY_ROOT)
set( UTILITY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/utility )
endif()
message(STATUS "UTILITY_ROOT: ${UTILITY_ROOT}")

find_package(spdlog REQUIRED)
message(STATUS "Using spdlog ${spdlog_VERSION}")

add_definitions(-DBoost_USE_STATIC_LIBS=ON)
find_package(Boost 1.65 REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
message(STATUS "Using Boost ${Boost_VERSION}")

include_directories(${UTILITY_ROOT})

#target_link_libraries(${PROJECT_NAME} spdlog::spdlog)
#target_link_libraries(${PROJECT_NAME} Boost::filesystem Boost::boostsystem)

