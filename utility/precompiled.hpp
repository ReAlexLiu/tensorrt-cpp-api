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

#ifndef PRECOMPILED_HPP
#define PRECOMPILED_HPP

#ifndef UNUSED
#define UNUSED(x) (void)x;
#endif

#ifndef SINGLE_TPL
#define SINGLE_TPL(className)                               \
private:                                                    \
    className()                                             \
    {                                                       \
        create();                                           \
    }                                                       \
    ~className()                                            \
    {                                                       \
        destory();                                          \
    }                                                       \
                                                            \
public:                                                     \
    className(const className&)                   = delete; \
    className&        operator=(const className&) = delete; \
    static className& getInstance()                         \
    {                                                       \
        static className _instance;                         \
        return _instance;                                   \
    }
#endif

#ifndef FOLDER_SEPS
#ifdef _WIN32
#define FOLDER_SEPS '\\'
#else
#define FOLDER_SEPS '/'
#endif
#endif

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

#endif // PRECOMPILED_HPP
