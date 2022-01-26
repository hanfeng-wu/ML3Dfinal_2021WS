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
#include <stdio.h>
#include "print.hpp"
#include "macros.hpp"
#include "stacktrace.hpp"

#ifndef FTB_IO_IMPL

auto read_entire_file(const char* filename) -> String;

#else // implementations

auto read_entire_file(const char* filename) -> String  {
    String ret;
    ret.data = nullptr;
    FILE *fp = fopen(filename, "rb");
    if (fp) {
        defer { fclose(fp); };
        /* Go to the end of the file. */
        if (fseek(fp, 0L, SEEK_END) == 0) {
            /* Get the size of the file. */
            ret.length = ftell(fp) + 1;
            if (ret.length == 0) {
                fputs("Empty file", stderr);
                return ret;
            }

            /* Go back to the start of the file. */
            if (fseek(fp, 0L, SEEK_SET) != 0) {
                panic("Error reading file");
                return ret;
            }

            /* Allocate our buffer to that size. */
            ret.data = (char*)calloc(ret.length+1, sizeof(char));

            /* Read the entire file into memory. */
            ret.length = fread(ret.data, sizeof(char), ret.length, fp);

            ret.data[ret.length] = '\0';
            if (ferror(fp) != 0) {
                panic("Error reading file");
            }
        }
    } else {
        panic("Cannot read file: %s", filename);
    }

    return ret;
}

#endif // FTB_IO_IMPL
