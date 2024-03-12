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

#ifndef PROFILER_H
#define PROFILER_H

// 模仿aix系统xlc编译器的-qfunctrace参数，但是需要手动在函数起始位置添加执行代码，性能不美丽，慎用，慎用

#if HAS_PROFILER
#include "xlogger.hpp"
namespace utility
{
class profiler
{
public:
    using param_t = typename std::conditional<std::is_reference<const char*>::value || std::is_pointer<const char*>::value, const char*, const char*&>::type;

    profiler(param_t filename, param_t funcname, int line, std::function<void(param_t, param_t, int)>&& enter,
             std::function<void(param_t, param_t, int, param_t)>&& leavel)
        : leavel_(std::move(leavel))
        , filename_(filename)
        , funcname_(funcname)
        , line_(line)
        , message_("")
    {
        enter(filename, funcname, line);
    }

    profiler(param_t filename, param_t funcname, int line, param_t message, std::function<void(param_t, param_t, int, param_t)>&& enter,
             std::function<void(param_t, param_t, int, param_t)>&& leavel)
        : leavel_(std::move(leavel))
        , filename_(filename)
        , funcname_(funcname)
        , line_(line)
        , message_(message)
    {
        enter(filename, funcname, line, message);
    }

    ~profiler() { leavel_(filename_, funcname_, line_, message_); }

private:
    profiler()                            = delete;
    profiler(const profiler&)             = delete;
    profiler& operator=(const profiler&)  = delete;
    profiler(const profiler&&)            = delete;
    profiler& operator=(const profiler&&) = delete;

private:
    std::function<void(param_t, param_t, int, param_t)> leavel_;
    param_t                                             filename_;
    param_t                                             funcname_;
    int                                                 line_;
    param_t                                             message_;
};
}

#define PROFILER()                                                                                                  \
   utility::profiler p(                                                                                             \
       __FILENAME__, __FUNCTION__, __LINE__,                                                                       \
       [](const char* filename, const char* funcname, int line) {                                                  \
           utility::xlogger::getInstance().log_(spdlog::level::trace, filename, funcname, line, "{} enter", funcname);  \
       },                                                                                                          \
       [](const char* filename, const char* funcname, int line, const char* message) {                             \
           UNUSED(message);                                                                                        \
           utility::xlogger::getInstance().log_(spdlog::level::trace, filename, funcname, line, "{} leavel", funcname); \
       });

#define PROFILER_MESSAGE(message)                                                                                           \
   utility::profiler p(                                                                                                     \
       __FILENAME__, __FUNCTION__, __LINE__, message,                                                                      \
       [](const char* filename, const char* funcname, int line, const char* msg) {                                         \
           utility::xlogger::getInstance().log_(spdlog::level::trace, filename, funcname, line, "enter {}", msg);  \
       },                                                                                                                  \
       [](const char* filename, const char* funcname, int line, const char* msg) {                                         \
           utility::xlogger::getInstance().log_(spdlog::level::trace, filename, funcname, line, "leavel {}", msg); \
       });
#else
#define PROFILER() do{}while(0)
#define PROFILER_MESSAGE(message)  do{}while(0)
#endif
#endif  // PROFILER_H
