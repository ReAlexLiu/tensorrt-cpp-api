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

# 添加版本号，用于部署后验证程序是否升级成功
function(generate_version major minor branch_name)
    string(TIMESTAMP COMPILE_TIME "%Y-%m-%dT%H:%M:%S")
    set(VERSION_MAJOR ${major})    # 一级版本号
    set(VERSION_MINOR ${minor})    # 二级版本号

    set(TARGET_NAME ${branch_name}) 

    execute_process(
        COMMAND git log -1 --format=%H
        OUTPUT_VARIABLE GIT_VERSION
    )

    execute_process(
        COMMAND git log -1 --pretty=format:%h
        OUTPUT_VARIABLE GIT_SHORT_VERSION
    )
    
    if (GIT_SHORT_VERSION STREQUAL "")
        string(TIMESTAMP GIT_SHORT_VERSION "%H%M%S")
    endif()

    execute_process(
        COMMAND git log -1 --pretty=format:%an
        OUTPUT_VARIABLE GIT_AUTHOR
    )

    if (GIT_AUTHOR STREQUAL "")
        set(GIT_AUTHOR "lucky.liu")
    endif()

    execute_process(
        COMMAND git rev-list --all --count
        OUTPUT_VARIABLE VERSION_LEVEL3
    )
    
    if (VERSION_LEVEL3 STREQUAL "")
        string(TIMESTAMP VERSION_LEVEL3 "%Y%m%d")
    endif()

    configure_file(
        "config.h.in"
        "../config.h"
    )
endfunction(generate_version)

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

