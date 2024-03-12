/*
* Copyright (c) 2018 L2Q All rights reserved.
*
* The Original Code and all software distributed under the License are
* distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
* EXPRESS OR IMPLIED, AND L2Q HEREBY DISCLAIMS ALL SUCH WARRANTIES,
* INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
* Please see the License for the specific language governing rights and
* limitations under the License.
*
* @Descripttion: Public Macro Definition
* @Author: l2q
* @Date: 2021/3/8 13:27
* @LastEditors: lucky
* @LastEditTime: 2023/4/7 8:15
 */

#ifndef XLOGGER_HPP
#define XLOGGER_HPP

#include "config.hpp"
#include "precompiled.hpp"
#include "utility.hpp"

#include <spdlog/async.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <string>

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#ifdef WIN32
#include <spdlog/sinks/msvc_sink.h>
#define __FILENAME__ __FILE__
#endif

#include <iostream>
#include <string>
#include <vector>

#define UNUSED(x) (void)x;

namespace utility
{
class xlogger
{
    SINGLE_TPL(xlogger);

private:
    void create()
    {
        config& _config = config::getInstance();

#ifndef _DEBUG
        level_ = static_cast<spdlog::level::level_enum>(_config.get("global.log.level", SPDLOG_LEVEL_TRACE));
#else
        level_ = static_cast<spdlog::level::level_enum>(_config.get("global.log.level", SPDLOG_LEVEL_TRACE));
#endif

        std::vector<spdlog::sink_ptr> _sinkList;

        if (_config.get("global.log.console", true))
        {
            auto _stdoutColorSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            _stdoutColorSink->set_level(level_);
            _stdoutColorSink->set_pattern("%^[%L][%m-%d %H:%M:%S.%e][%!,%@][%t] %v%$");
            _sinkList.push_back(_stdoutColorSink);
#ifdef WIN32
            auto _msvcColorSink = std::make_shared<spdlog::sinks::windebug_sink_mt>();
            _msvcColorSink->set_level(level_);
            _msvcColorSink->set_pattern("%^[%L][%m-%d %H:%M:%S.%e][%!,%@][%t] %v%$");
            _sinkList.push_back(_msvcColorSink);
#endif  // WIN32
        }

        std::string _save_path = utility::basepath();
        _save_path += "/../logs/";
        _save_path += utility::basename();
        _save_path += ".log";


        std::size_t _max_files = _config.get("global.log.max_files", 7);
        if (_config.get("global.log.file", false))
        {
            std::size_t _max_size        = _config.get("global.log.max_size", 200) * 1024 * 1024;  // 默认200M
            auto        rotatingFileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(_save_path, _max_size, _max_files);
            rotatingFileSink->set_level(level_);
            rotatingFileSink->set_pattern("[%L][%Y-%m-%d %H:%M:%S.%e][%!,%@][%t] %v");
            _sinkList.push_back(rotatingFileSink);
        }
        else
        {
            int  _rotation_hour   = _config.get("global.log.rotation_hour", 0);
            int  _rotation_minute = _config.get("global.log.rotation_minute", 1);
            auto dailyFileSink    = std::make_shared<spdlog::sinks::daily_file_sink_mt>(_save_path, _rotation_hour, _rotation_minute, false, static_cast<int>(_max_files * 4));
            // auto dailyFileSink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(_save_path, _rotation_hour, _rotation_minute);
            dailyFileSink->set_level(level_);
            dailyFileSink->set_pattern("[%L][%Y-%m-%d %H:%M:%S.%e][%!,%@][%t] %v");
            _sinkList.push_back(dailyFileSink);
        }

        if (_sinkList.size() > 0)
        {
            // async
            // spdlog::init_thread_pool(2048, 1);
            // logger_ = std::make_shared<spdlog::async_logger>("multi_sink", _sinkList.begin(), _sinkList.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            // spdlog::register_logger(logger_);
            // logger_->flush_on(spdlog::level::err);
            // spdlog::flush_every(std::chrono::seconds(_config.get("global.log.interval", 10)));
            // logger_->set_level(level_);

            // sync
            logger_ = std::make_shared<spdlog::logger>("multi_sink", begin(_sinkList), end(_sinkList));
            spdlog::register_logger(logger_);
            logger_->flush_on(spdlog::level::critical);

            spdlog::flush_every(std::chrono::seconds(_config.get("global.log.interval", 10)));
            logger_->set_level(level_);
        }
    }

    void destory()
    {
        //  spdlog::drop_all();
    }

public:
    void setLevel(int level)
    {
        level_ = static_cast<spdlog::level::level_enum>(level);
        logger_->set_level(level_);
    }

    template <typename... Args>
    void log_(spdlog::level::level_enum lvl, const char* filename, const char* funcname, int line, spdlog::string_view_t fmt, const Args&... args)
    {
        try
        {
            spdlog::source_loc loc(filename, line, funcname);
            logger_->log(loc, lvl, fmt, args...);
        }
        catch (...)
        {
            std::cerr << "failed to write log" << std::endl;
        }
    }

private:
    spdlog::level::level_enum       level_;
    // std::shared_ptr<spdlog::async_logger> logger_;
    std::shared_ptr<spdlog::logger> logger_;
};
}

// 基础的输出封装
#define xlevel(level) utility::xlogger::getInstance().setLevel(level)
#define xverbose(...) utility::xlogger::getInstance().log_(spdlog::level::trace, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define xdebug(...) utility::xlogger::getInstance().log_(spdlog::level::debug, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define xinfo(...) utility::xlogger::getInstance().log_(spdlog::level::info, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define xwarn(...) utility::xlogger::getInstance().log_(spdlog::level::warn, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define xerror(...) utility::xlogger::getInstance().log_(spdlog::level::err, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define xfatal(...) utility::xlogger::getInstance().log_(spdlog::level::critical, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define xverbose_if(exp, ...)                                                                                                       \
    if (exp)                                                                                                                        \
    {                                                                                                                               \
        utility::xlogger::getInstance().log_(spdlog::level::trace, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__); \
    }

#define xdebug_if(exp, ...)                                                                                                         \
    if (exp)                                                                                                                        \
    {                                                                                                                               \
        utility::xlogger::getInstance().log_(spdlog::level::debug, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__); \
    }

#define xinfo_if(exp, ...)                                                                                                         \
    if (exp)                                                                                                                       \
    {                                                                                                                              \
        utility::xlogger::getInstance().log_(spdlog::level::info, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__); \
    }

#define xwarn_if(exp, ...)                                                                                                         \
    if (exp)                                                                                                                       \
    {                                                                                                                              \
        utility::xlogger::getInstance().log_(spdlog::level::warn, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__); \
    }

#define xerror_if(exp, ...)                                                                                                       \
    if (exp)                                                                                                                      \
    {                                                                                                                             \
        utility::xlogger::getInstance().log_(spdlog::level::err, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__); \
    }

#define xfatal_if(exp, ...)                                                                                                            \
    if (exp)                                                                                                                           \
    {                                                                                                                                  \
        utility::xlogger::getInstance().log_(spdlog::level::critical, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__); \
    }
#endif  // XLOGGER_HPP
