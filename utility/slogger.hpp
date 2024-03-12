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

#ifndef SLOGGER_HPP
#define SLOGGER_HPP
#include <sstream>

#include "xlogger.hpp"

namespace utility
{
class logstream
{
public:
    logstream(spdlog::level::level_enum lvl, const char* filename, const char* funcname, int line)
        : level_(lvl)
        , filename_(filename)
        , funcname_(funcname)
        , line_(line)
    {
    }

    ~logstream()
    {
        xlogger::getInstance().log_(level_, filename_.c_str(), funcname_.c_str(), line_, stream_.str().c_str());
    }

    std::ostringstream& log_()
    {
        return stream_;
    }

    static void setLevel(const std::string &log_level)
    {
        if(log_level == "DEBUG")
        {
            xlogger::getInstance().setLevel(1);
        }
        else if(log_level == "INFO")
        {
            xlogger::getInstance().setLevel(2);
        }
        else if(log_level == "WARN")
        {
            xlogger::getInstance().setLevel(3);
        }
        else if(log_level == "ERROR")
        {
            xlogger::getInstance().setLevel(4);
        }
        else if(log_level == "FATAL")
        {
            xlogger::getInstance().setLevel(5);
        }
        else if(log_level == "NONE")
        {
            xlogger::getInstance().setLevel(6);
        }
    }

private:
    std::ostringstream        stream_;
    spdlog::level::level_enum level_;
    std::string               filename_;
    std::string               funcname_;
    int                       line_;
};
}

#define LogLevel(level) utility::logstream::setLevel(level)
#define LogDebug utility::logstream(spdlog::level::debug, __FILENAME__, __FUNCTION__, __LINE__).log_()
#define LogInfo utility::logstream(spdlog::level::info, __FILENAME__, __FUNCTION__, __LINE__).log_()
#define LogWarn utility::logstream(spdlog::level::warn, __FILENAME__, __FUNCTION__, __LINE__).log_()
#define LogError utility::logstream(spdlog::level::err, __FILENAME__, __FUNCTION__, __LINE__).log_()
#define LogFatal utility::logstream(spdlog::level::critical, __FILENAME__, __FUNCTION__, __LINE__).log_()
#endif