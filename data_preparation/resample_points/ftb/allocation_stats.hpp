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
#include <stdlib.h>
#include <stdio.h>

#include "types.hpp"
#include "macros.hpp"

#ifdef FTB_TRACK_MALLOCS
namespace Ftb_Malloc_Stats {
    extern u32 malloc_calls;
    extern u32 free_calls;
    extern u32 realloc_calls;
    extern u32 calloc_calls;
}

#define malloc(size)       (++Ftb_Malloc_Stats::malloc_calls,  malloc(size))
#define free(ptr)          (++Ftb_Malloc_Stats::free_calls,    free(ptr))
#define realloc(ptr, size) (++Ftb_Malloc_Stats::realloc_calls, realloc(ptr, size))
#define calloc(num, size)  (++Ftb_Malloc_Stats::calloc_calls,  calloc(num, size))

#define profile_mallocs                                                 \
    MPP_DECLARE(0, u32 MPI_LABEL(profile_mallocs, old_malloc_calls)  = Ftb_Malloc_Stats::malloc_calls) \
    MPP_DECLARE(1, u32 MPI_LABEL(profile_mallocs, old_free_calls)    = Ftb_Malloc_Stats::free_calls) \
    MPP_DECLARE(2, u32 MPI_LABEL(profile_mallocs, old_realloc_calls) = Ftb_Malloc_Stats::realloc_calls) \
    MPP_DECLARE(3, u32 MPI_LABEL(profile_mallocs, old_calloc_calls)  = Ftb_Malloc_Stats::calloc_calls) \
    MPP_AFTER(4, {                                                      \
            printf("\n"                                                 \
                   "Local Malloc Stats: (%s %s %d)\n"                   \
                   "-------------------\n"                              \
                   "   malloc calls: %u\n"                              \
                   "     free calls: %u\n"                              \
                   "  realloc calls: %u\n"                              \
                   "   calloc calls: %u\n",                             \
                   __func__, __FILE__, __LINE__,                        \
                   Ftb_Malloc_Stats::malloc_calls  - MPI_LABEL(profile_mallocs, old_malloc_calls)  , \
                   Ftb_Malloc_Stats::free_calls    - MPI_LABEL(profile_mallocs, old_free_calls)    , \
                   Ftb_Malloc_Stats::realloc_calls - MPI_LABEL(profile_mallocs, old_realloc_calls) , \
                   Ftb_Malloc_Stats::calloc_calls  - MPI_LABEL(profile_mallocs, old_calloc_calls));  \
        })
#endif // FTB_TRACK_MALLOCS

#ifndef FTB_ALLOCATION_STATS_IMPL

auto print_malloc_stats() -> void;

#else

namespace Ftb_Malloc_Stats {
    u32 malloc_calls  = 0;
    u32 free_calls    = 0;
    u32 realloc_calls = 0;
    u32 calloc_calls  = 0;
}

auto print_malloc_stats() -> void {
    printf("\n"
           "Global Malloc Stats:\n"
           "--------------------\n"
           "   malloc calls: %u\n"
           "     free calls: %u\n"
           "  realloc calls: %u\n"
           "   calloc calls: %u\n",
           Ftb_Malloc_Stats::malloc_calls,
           Ftb_Malloc_Stats::free_calls,
           Ftb_Malloc_Stats::realloc_calls,
           Ftb_Malloc_Stats::calloc_calls);
}

#endif //FTB_ALLOCATION_STATS_IMPL
