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
#include "allocation_stats.hpp"

#ifndef FTB_STACKTRACE_IMPL

auto print_stacktrace() -> void;

#else // implementations

#  ifndef FTB_STACKTRACE_INFO

auto print_stacktrace() -> void {
    printf("No stacktrace info available (recompile with FTB_STACKTRACE_INFO defined)\n");
}

#  else // stacktace should be present
#    if defined FTB_WINDOWS
#      include <Windows.h>
#      include <dbghelp.h>

auto print_stacktrace() -> void {
    printf("Stacktrace: \n");
    unsigned int   i;
    void         * stack[ 100 ];
    HANDLE         process;
    SYMBOL_INFO  * symbol;
    symbol               = ( SYMBOL_INFO * )calloc( sizeof( SYMBOL_INFO ) + 256 * sizeof( char ), 1 );
    symbol->MaxNameLen   = 255;
    symbol->SizeOfStruct = sizeof( SYMBOL_INFO );

    unsigned short frames;
    frames = CaptureStackBackTrace( 1, 100, stack, NULL );

    process = GetCurrentProcess();
    SymInitialize( process, NULL, TRUE );

    for( i = 0; i < frames; i++ ) {
        SymFromAddr( process, ( DWORD64 )( stack[ i ] ), 0, symbol );
        printf( "  %3i: %s\n", frames - i - 1, symbol->Name);
    }
    fflush(stdout);
}

#    else // not windows
#      include <stdio.h>
#      include <stdlib.h>
#      include <unistd.h>
#      ifdef FTB_STACKTRACE_USE_GDB
#        include <sys/wait.h>
#        include <sys/prctl.h>

auto print_stacktrace() -> void {
    printf("Stacktrace: \n");
    char pid_buf[30];
    sprintf(pid_buf, "%d", getpid());
    char name_buf[512];
    name_buf[readlink("/proc/self/exe", name_buf, 511)]=0;
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    int child_pid = fork();
    if (!child_pid) {
        dup2(2,1); // redirect output to stderr - edit: unnecessary?
        execl("/usr/bin/gdb", "gdb", "--batch", "-n", "-ex", "thread", "-ex", "bt", name_buf, pid_buf, NULL);
        abort(); /* If gdb failed to start */
    } else {
        waitpid(child_pid,NULL,0);
   }
}

#      else // linux, but not using gdb
#        include <execinfo.h>

auto print_stacktrace() -> void {
    printf("Stacktrace (this is unmagled -- sorry\n"
           "  (you can recompile with FTB_STACKTRACE_USE_GDB defined and -rdynamic to get one) \n");
    char **strings;
    size_t i, size;
    enum Constexpr { MAX_SIZE = 1024 };
    void *array[MAX_SIZE];
    size = backtrace(array, MAX_SIZE);
    strings = backtrace_symbols(array, size);
    for (i = 0; i < size; i++)
        printf("  %3lu: %s\n", size - i - 1, strings[i]);
    puts("");
    free(strings);
}

#      endif // FTB_STACKTRACE_USE_GDB
#    endif   // FTB_WINDOWS
#  endif     // FTB_STACKTRACE_INFO
#endif       // FTB_STACKTRACE_IMPL
