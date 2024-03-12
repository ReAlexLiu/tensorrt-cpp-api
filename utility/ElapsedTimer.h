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

#ifndef ELAPSEDTIMER_H
#define ELAPSEDTIMER_H

#include <chrono>
namespace utility
{
class ElapsedTimer
{
public:
    ElapsedTimer()
        : t1(INT64_C(0x8000000000000000))
    {
        start();
    }

    void start()
    {
        restart();
    }

    uint64_t restart()
    {
        t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        return t1;
    }

    int64_t elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1;
    }

    bool hasExpired(int64_t timeout) const noexcept
    {
        // if timeout is -1, quint64(timeout) is LLINT_MAX, so this will be
        // considered as never expired
        return elapsed() > timeout;
    }

private:
    int64_t t1;
};
}

#endif  // ELAPSEDTIMER_H
