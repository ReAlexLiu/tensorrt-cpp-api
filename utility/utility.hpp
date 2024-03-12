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

#ifndef UTILS_HPP
#define UTILS_HPP

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "precompiled.hpp"

#include <string>
#include <sstream>
#include <string.h>
#ifdef _WIN32
#include <direct.h> 
#include <io.h>
#ifdef DYNAMIC_LINK_LIBRARY
HMODULE GetSelfModuleHandle()
{
    MEMORY_BASIC_INFORMATION mbi;
    return ((::VirtualQuery(GetSelfModuleHandle, &mbi, sizeof(mbi)) != 0) ? (HMODULE)mbi.AllocationBase : NULL);
}
#else
#include <windows.h>
#endif
#else
#include <unistd.h>
#include <stdarg.h> 
#include <sys/stat.h> 
#ifdef DYNAMIC_LINK_LIBRARY
#include <dlfcn.h>
void srdladdr(){}
#else
#include <unistd.h>
#endif
#endif

namespace utility
{
class utility
{
public:
    static std::string basepath()
    {
        static std::string _dir_path;
        if (_dir_path.empty())
        {
            char  _temp[MAX_PATH] = { 0 };
            int   _size           = -1;
            char* _buffer         = _temp;

#ifdef _WIN32
#ifdef DYNAMIC_LINK_LIBRARY
            _size = GetModuleFileNameA(GetSelfModuleHandle(), _temp, MAX_PATH);
#else
            _size = GetModuleFileNameA(nullptr, _temp, MAX_PATH);
#endif
            if (_size > MAX_PATH - 1)
            {
                _buffer = new (std::nothrow) char[_size + 1]();
                if (_buffer)
                {
#ifdef DYNAMIC_LINK_LIBRARY
                    GetModuleFileNameA(GetSelfModuleHandle(), _buffer, _size + 1);
#else
                    GetModuleFileNameA(nullptr, _buffer, _size + 1);
#endif
                }
            }
#else
#ifdef DYNAMIC_LINK_LIBRARY
            Dl_info _dl_info;
            dladdr((void*)srdladdr, &_dl_info);
            _size = strlen(_dl_info.dli_fname);
            if (_size > MAX_PATH - 1)
            {
                _buffer = new (std::nothrow) char[_size + 1]();
                if (_buffer)
                {
                    strcpy(_buffer, _dl_info.dli_fname);
                }
            }
            else
            {
                strcpy(_temp, _dl_info.dli_fname);
            }
#else
            _size = readlink("/proc/self/exe", _temp, MAX_PATH);
            if (_size > MAX_PATH - 1)
            {
                _buffer = new (std::nothrow) char[_size + 1]();
                if (_buffer)
                {
                    readlink("/proc/self/exe", _buffer, _size + 1);
                }
            }
#endif
#endif
            *strrchr(_buffer, FOLDER_SEPS) = '\0';
            _dir_path = std::string(_buffer);
        }
        return _dir_path;
    }

    static bool create_dir(const std::string &strPath)
    {
        std::string strTempPath1 = strPath;
        size_t index = strTempPath1.find(FOLDER_SEPS);
        std::string strTempPath2 = strTempPath1.substr(0, index);
        strTempPath1 = strTempPath1.substr(index, strTempPath1.size());

        while (index >= 0)
        {
            index = strTempPath1.find(FOLDER_SEPS);

            strTempPath2 += FOLDER_SEPS + strTempPath1.substr(0, index);
            //判断该目录是否存在
#ifdef _WIN32
            if (-1 == _access(strTempPath2.c_str(), 0))
            {
                if (_mkdir(strTempPath2.c_str()) != 0)     //创建目录
#else
            if (-1 == access(strTempPath2.c_str(), 0))
            {
                if (mkdir(strTempPath2.c_str(), 0755) != 0)     //创建目录
#endif
                {
                    return false;
                }
            }

            strTempPath1 = strTempPath1.substr(index + 1, strTempPath1.size());
        }

        return true;
    }

    static inline bool p_strchr(const char str[], char chr)
    {
        if (!str) return false;

        const char* ptr = str;
        char c;
        while ((c = *ptr++))
            if (c == chr)
                return true;
        return false;
    }

    static std::string basename()
    {
        static std::string _file_name;
        if (_file_name.empty())
        {
            char  _temp[MAX_PATH] = { 0 };
            int   _size           = -1;
            char* _buffer         = _temp;

#ifdef _WIN32
#ifdef DYNAMIC_LINK_LIBRARY
            _size = GetModuleFileNameA(GetSelfModuleHandle(), _temp, MAX_PATH);
#else
            _size = GetModuleFileNameA(nullptr, _temp, MAX_PATH);
#endif
            if (_size > MAX_PATH - 1)
            {
                _buffer = new (std::nothrow) char[_size + 1]();
                if (_buffer)
                {
#ifdef DYNAMIC_LINK_LIBRARY
                    GetModuleFileNameA(GetSelfModuleHandle(), _buffer, _size + 1);
#else
                    GetModuleFileNameA(nullptr, _buffer, _size + 1);
#endif
                }
            }
#else
#ifdef DYNAMIC_LINK_LIBRARY
            Dl_info _dl_info;
            dladdr((void*)srdladdr, &_dl_info);
            _size = strlen(_dl_info.dli_fname);
            if (_size > MAX_PATH - 1)
            {
                _buffer = new (std::nothrow) char[_size + 1]();
                if (_buffer)
                {
                    strcpy(_buffer, _dl_info.dli_fname);
                }
            }
            else
            {
                strcpy(_temp, _dl_info.dli_fname);
            }
#else
            _size = readlink("/proc/self/exe", _temp, MAX_PATH);
            if (_size > MAX_PATH - 1)
            {
                _buffer = new (std::nothrow) char[_size + 1]();
                if (_buffer)
                {
                    readlink("/proc/self/exe", _buffer, _size + 1);
                }
            }
#endif
#endif
            _file_name = std::string(strrchr(_buffer, FOLDER_SEPS) +1);
        }
        return _file_name;
    }

    static std::string percent_encoding(const std::string &input, const char* dontEncode = nullptr, const char* alsoEncode = nullptr, char percent = '%')
    {
        const char* _input = input.c_str();
        std::stringstream _output;

        for (unsigned int i = 0; i < input.size(); ++i) {
            unsigned char c = *_input++;
            if (((c >= 0x61 && c <= 0x7A) // ALPHA
                || (c >= 0x41 && c <= 0x5A) // ALPHA
                || (c >= 0x30 && c <= 0x39) // DIGIT
                || c == 0x2D // -
                || c == 0x2E // .
                || c == 0x5F // _
                || c == 0x7E // ~
                || p_strchr(dontEncode, c))
                && !p_strchr(alsoEncode, c)) {
                _output << c;
            }
            else 
            {
                _output << percent << std::uppercase << std::hex << ((c & 0xf0) >> 4) << (c & 0xf);
            }
        }
        return _output.str();
    }

    static std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str)
    {
        std::string dst_str = resource_str;
        std::string::size_type pos = 0;
        while((pos = dst_str.find(sub_str)) != std::string::npos)   //替换所有指定子串
        {
            dst_str.replace(pos, sub_str.length(), new_str);
        }
        return dst_str;
    }
};
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif  // UTILS_HPP

