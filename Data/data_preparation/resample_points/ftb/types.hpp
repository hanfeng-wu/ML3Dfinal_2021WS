/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Felix Brendel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "platform.hpp"
#include <stdint.h>
#include <string.h>

typedef int8_t   s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t   u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef uint8_t byte;

typedef float       f32;
typedef long double f64;

#ifdef UNICODE
typedef wchar_t path_char;
#else
typedef char    path_char;
#endif

struct String_Slice {
    const char* data;
    u64 length;
};

struct String {
    char* data;
    u64 length;
};

#ifndef FTB_TYPES_IMPL

inline auto heap_copy_c_string(const char* str) -> char*;
inline auto make_heap_string(const char* str) -> String;
inline auto make_static_string(const char* str) -> const String_Slice;
inline auto string_equal(const char* input, const char* check) -> bool;
inline auto string_equal(String_Slice str, const char* check) -> bool;
inline auto string_equal(const char* check, String_Slice str) -> bool;
inline auto string_equal(String_Slice str1, String_Slice str2) -> bool;

#else // implementations

inline auto heap_copy_c_string(const char* str) -> char* {
#ifdef FTB_WINDOWS
    return _strdup(str);
#else
    return strdup(str);
#endif
}

inline auto make_heap_string(const char* str) -> String {
    String ret;
    ret.length = strlen(str);
    ret.data = heap_copy_c_string(str);
    return ret;
}

inline auto make_static_string(const char* str) -> const String_Slice {
    String_Slice ret;
    ret.length = strlen(str);
    ret.data = str;
    return ret;
}

auto inline string_equal(const char* input, const char* check) -> bool {
    return strcmp(input, check) == 0;
}

auto inline string_equal(String_Slice str, const char* check) -> bool {
    if (str.length != strlen(check))
        return false;
    return strncmp(str.data, check, str.length) == 0;
}

auto inline string_equal(const char* check, String_Slice str) -> bool {
    if (str.length != strlen(check))
        return false;
    return strncmp(str.data, check, str.length) == 0;
}

auto inline string_equal(String_Slice str1, String_Slice str2) -> bool {
    if (str1.length != str2.length)
        return false;

    return strncmp(str1.data, str2.data, str2.length) == 0;
}

#endif // FTB_TYPES_IMPL
