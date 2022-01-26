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

// The MPI_ and MPP_ macros are taken from Simon Tatham
// https://www.chiark.greenend.org.uk/~sgtatham/mp/mp.h
/*
 * mp.h is copyright 2012 Simon Tatham.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL SIMON TATHAM BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * $Id$
 */

#pragma once
#include "platform.hpp"

#define array_length(arr) (sizeof(arr) / sizeof(arr[0]))

#ifdef FTB_WINDOWS
#else
#  include <signal.h> // for sigtrap
#endif

#define CONCAT(x, y) x ## y
#define LABEL(x, y) CONCAT(x, y)

#define MPI_LABEL(id1,id2)                              \
    LABEL(MPI_LABEL_ ## id1 ## _ ## id2 ## _, __LINE__)

#define MPP_DECLARE(labid, declaration)                 \
    if (0)                                              \
        ;                                               \
    else                                                \
        for (declaration;;)                             \
            if (1) {                                    \
                goto MPI_LABEL(labid, body);            \
              MPI_LABEL(labid, done): break;            \
            } else                                      \
                while (1)                               \
                    if (1)                              \
                        goto MPI_LABEL(labid, done);    \
                    else                                \
                        MPI_LABEL(labid, body):

#define MPP_BEFORE(labid,before)                \
    if (1) {                                    \
        before;                                 \
        goto MPI_LABEL(labid, body);            \
    } else                                      \
    MPI_LABEL(labid, body):

#define MPP_AFTER(labid,after)                  \
    if (1)                                      \
        goto MPI_LABEL(labid, body);            \
    else                                        \
        while (1)                               \
            if (1) {                            \
                after;                          \
                break;                          \
            } else                              \
                MPI_LABEL(labid, body):


/**
 *   Defer   *
 */
#ifndef defer
struct defer_dummy {};
template <class F> struct deferrer { F f; ~deferrer() { f(); } operator bool() const { return false; } };
template <class F> deferrer<F> operator*(defer_dummy, F f) { return {f}; }
#define DEFER_(LINE) zz_defer##LINE
#define DEFER(LINE) DEFER_(LINE)
#define defer auto DEFER(__LINE__) = defer_dummy{} *[&]()
#endif // defer

#define defer_free(var) defer { free(var); }

/*
   defer {
       call();
    };

expands to:

    auto zz_defer74 = defer_dummy{} * [&] {
       call();
    };
*/

#if defined(unix) || defined(__unix__) || defined(__unix)
#define NULL_HANDLE "/dev/null"
#else
#define NULL_HANDLE "nul"
#endif
#define ignore_stdout                                                   \
    if (0)                                                              \
        LABEL(finished,__LINE__): ;                                     \
    else                                                                \
        for (FILE* LABEL(fluid_let_, __LINE__) = ftb_stdout;;)                             \
            for (defer{ fclose(ftb_stdout); ftb_stdout=LABEL(fluid_let_, __LINE__) ; } ;;) \
                if (1) {                                                \
                    ftb_stdout = fopen(NULL_HANDLE, "w");               \
                    goto LABEL(body,__LINE__);                          \
                }                                                       \
                else                                                    \
                    while (1)                                           \
                        if (1) {                                        \
                            goto LABEL(finished, __LINE__);             \
                        }                                               \
                        else LABEL(body,__LINE__):


/*****************
 *   fluid-let   *
 *****************/

#define fluid_let(var, val)                                             \
    if (0)                                                              \
        LABEL(finished,__LINE__): ;                               \
    else                                                                \
        for (auto LABEL(fluid_let_, __LINE__) = var;;)            \
            for (defer{var = LABEL(fluid_let_, __LINE__);};;)     \
                for(var = val;;)                                        \
                    if (1) {                                            \
                        goto LABEL(body,__LINE__);                \
                    }                                                   \
                    else                                                \
                        while (1)                                       \
                            if (1) {                                    \
                                goto TOKENPASTE2(finished, __LINE__);   \
                            }                                           \
                            else TOKENPASTE2(body,__LINE__):
                                     ;


/**
fluid_let(var, val) {
    call1(var);
    call2(var);
}

expands to

if (0)
    finished98:;
else
    for (auto fluid_let_98 = var;;)
        for (auto __deferred_lambda_call0 = deferrer << [&] { var = fluid_let_98; };;)
            for (var = val;;)
                if (1) {
                    goto body98;
                } else
                    while (1)
                        if (1) {
                            goto finished98;
                        } else
                          body98 : {
                              call1(var);
                              call2(var);
                          }
*/

#define panic(...)                                                          \
    do {                                                                    \
        print("%{color<}[Panic]%{>color} in "                               \
              "file %{color<}%{->char}%{>color} "                           \
              "line %{color<}%{u32}%{>color}: "                             \
              "(%{color<}%{->char}%{>color})\n",                            \
              console_red, console_cyan, __FILE__, console_cyan, __LINE__,  \
              console_cyan, __func__);                                      \
        print("%{color<}", console_red);                                    \
        print(__VA_ARGS__);                                                 \
        print("%{>color}\n");                                               \
        fflush(stdout);                                                     \
        print_stacktrace();                                                 \
        debug_break();                                                      \
    } while(0)


#define panic_if(cond, ...)                     \
    if(!(cond));                                \
    else panic(__VA_ARGS__)

#ifdef FTB_DEBUG_LOG
#  define debug_log(...)                                        \
    do {                                                        \
        print("%{color<}[INFO " __FILE__ ":%{color<}%{->char}"  \
              "%{>color}]%{>color} ",                           \
              console_green, console_cyan, __func__);           \
        println(__VA_ARGS__);                                   \
    } while (0)
#else
#  define debug_log(...)
#endif

#ifdef FTB_WINDOWS
#  define debug_break() __debugbreak()
#else
#  define debug_break() raise(SIGTRAP);
#endif
