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
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>

#ifdef FTB_WINDOWS
#  include <Windows.h>
#endif

#include "hashmap.hpp"

extern FILE* ftb_stdout;

// NOTE(Felix): These are defines, so that the preprocessor can concat string
//   literals as in: `console_red "hello" console_normal'
#define console_red     "\x1B[31m"
#define console_green   "\x1B[32m"
#define console_yellow  "\x1B[33m"
#define console_blue    "\x1B[34m"
#define console_magenta "\x1B[35m"
#define console_cyan    "\x1B[36m"
#define console_white   "\x1B[37m"
#define console_normal  "\x1B[0m"

typedef const char* static_string;
typedef int (*printer_function_32b)(FILE*, u32);
typedef int (*printer_function_64b)(FILE*, u64);
typedef int (*printer_function_flt)(FILE*, double);
typedef int (*printer_function_ptr)(FILE*, void*);
typedef int (*printer_function_void)(FILE*);

enum struct Printer_Function_Type {
    unknown,
    _32b,
    _64b,
    _flt,
    _ptr,
    _void
};

#define register_printer(spec, fun, type)                       \
    register_printer_ptr(spec, (printer_function_ptr)fun, type)

#ifndef FTB_PRINT_IMPL

auto register_printer_ptr(const char* spec, printer_function_ptr fun, Printer_Function_Type type) -> void;
auto print_va_args_to_file(FILE* file, static_string format, va_list* arg_list) -> s32;
auto print_va_args_to_string(char** out, static_string format, va_list* arg_list)  -> s32;
auto print_va_args(static_string format, va_list* arg_list) -> s32;
auto print_to_string(char** out, static_string format, ...) -> s32;
auto print_to_file(FILE* file, static_string format, ...) -> s32;
auto print(static_string format, ...) -> s32;
auto println(static_string format, ...) -> s32;
auto init_printer() -> void;
auto deinit_printer() -> void;

#else // implementations
FILE* ftb_stdout = stdout;

Array_List<char*>                     color_stack = {};
Hash_Map<char*, printer_function_ptr> printer_map = {};
Hash_Map<char*, int>                  type_map    = {};

void register_printer_ptr(const char* spec, printer_function_ptr fun, Printer_Function_Type type) {
    printer_map.set_object((char*)spec, fun);
    type_map.set_object((char*)spec, (int)type);
}

int maybe_special_print(FILE* file, static_string format, int* pos, va_list* arg_list) {
    if(format[*pos] != '{')
        return 0;

    int end_pos = (*pos)+1;
    while (format[end_pos] != '}' &&
           format[end_pos] != ',' &&
           format[end_pos] != '[' &&
           format[end_pos] != 0)
        ++end_pos;

    if (format[end_pos] == 0)
        return 0;

    char* spec = (char*)alloca(end_pos - (*pos));
    strncpy(spec, format+(*pos)+1, end_pos - (*pos));
    spec[end_pos - (*pos)-1] = '\0';

    u64 spec_hash = hm_hash(spec);
    Printer_Function_Type type = (Printer_Function_Type)type_map.get_object(spec, spec_hash);


    union {
        printer_function_32b printer_32b;
        printer_function_64b printer_64b;
        printer_function_ptr printer_ptr;
        printer_function_flt printer_flt;
        printer_function_void printer_void;
    } printer;

    // just grab it, it will have the correct type
    printer.printer_ptr = printer_map.get_object(spec, spec_hash);

    if (type == Printer_Function_Type::unknown) {
        printf("ERROR: %s printer not found\n", spec);
        fflush(stdout);
        return 0;
    }

    if (format[end_pos] == ',') {
        int element_count;

        ++end_pos;
        sscanf(format+end_pos, "%d", &element_count);

        while (format[end_pos] != '}' &&
               format[end_pos] != 0)
            ++end_pos;
        if (format[end_pos] == 0)
            return 0;

        // both brackets already included:
        int written_length = 2;

        fputs("[", file);
        for (int i = 0; i < element_count - 1; ++i) {
            if      (type == Printer_Function_Type::_32b) written_length += printer.printer_32b(file, va_arg(*arg_list, u32));
            else if (type == Printer_Function_Type::_64b) written_length += printer.printer_64b(file, va_arg(*arg_list, u64));
            else if (type == Printer_Function_Type::_flt) written_length += printer.printer_flt(file, va_arg(*arg_list, double));
            else if (type == Printer_Function_Type::_ptr) written_length += printer.printer_ptr(file, va_arg(*arg_list, void*));
            else                                          written_length += printer.printer_void(file);
            written_length += 2;
            fputs(", ", file);
        }
        if (element_count > 0) {
            if      (type == Printer_Function_Type::_32b) written_length += printer.printer_32b(file, va_arg(*arg_list, u32));
            else if (type == Printer_Function_Type::_64b) written_length += printer.printer_64b(file, va_arg(*arg_list, u64));
            else if (type == Printer_Function_Type::_flt) written_length += printer.printer_flt(file, va_arg(*arg_list, double));
            else if (type == Printer_Function_Type::_ptr) written_length += printer.printer_ptr(file, va_arg(*arg_list, void*));
            else                                          written_length += printer.printer_void(file);
        }
        fputs("]", file);

        *pos = end_pos;
        return written_length;
    } else if (format[end_pos] == '[') {
        end_pos++;
        u32 element_count;
        union {
            u32*    arr_32b;
            u64*    arr_64b;
            f32*    arr_flt;
            void**  arr_ptr;
        } arr;

        if      (type == Printer_Function_Type::_32b) arr.arr_32b = va_arg(*arg_list, u32*);
        else if (type == Printer_Function_Type::_64b) arr.arr_64b = va_arg(*arg_list, u64*);
        else if (type == Printer_Function_Type::_flt) arr.arr_flt = va_arg(*arg_list, f32*);
        else                                          arr.arr_ptr = va_arg(*arg_list, void**);

        if (format[end_pos] == '*') {
            element_count =  va_arg(*arg_list, u32);
        } else {
            sscanf(format+end_pos, "%d", &element_count);
        }

        // skip to next ']'
        while (format[end_pos] != ']' &&
               format[end_pos] != 0)
            ++end_pos;
        if (format[end_pos] == 0)
            return 0;

        // skip to next '}'
        while (format[end_pos] != '}' &&
               format[end_pos] != 0)
            ++end_pos;
        if (format[end_pos] == 0)
            return 0;

        // both brackets already included:
        int written_length = 2;

        fputs("[", file);
        for (u32 i = 0; i < element_count - 1; ++i) {
            if      (type == Printer_Function_Type::_32b) written_length += printer.printer_32b(file, arr.arr_32b[i]);
            else if (type == Printer_Function_Type::_64b) written_length += printer.printer_64b(file, arr.arr_64b[i]);
            else if (type == Printer_Function_Type::_flt) written_length += printer.printer_flt(file, arr.arr_flt[i]);
            else if (type == Printer_Function_Type::_ptr) written_length += printer.printer_ptr(file, arr.arr_ptr[i]);
            else                                          written_length += printer.printer_void(file);
            written_length += 2;
            fputs(", ", file);
        }
        if (element_count > 0) {
            if      (type == Printer_Function_Type::_32b) written_length += printer.printer_32b(file, arr.arr_32b[element_count - 1]);
            else if (type == Printer_Function_Type::_64b) written_length += printer.printer_64b(file, arr.arr_64b[element_count - 1]);
            else if (type == Printer_Function_Type::_flt) written_length += printer.printer_flt(file, arr.arr_flt[element_count - 1]);
            else if (type == Printer_Function_Type::_ptr) written_length += printer.printer_ptr(file, arr.arr_ptr[element_count - 1]);
            else                                          written_length += printer.printer_void(file);
        }
        fputs("]", file);

        *pos = end_pos;
        return written_length;
    } else {
        *pos = end_pos;
        if      (type == Printer_Function_Type::_32b) return printer.printer_32b(file, va_arg(*arg_list, u32));
        else if (type == Printer_Function_Type::_64b) return printer.printer_64b(file, va_arg(*arg_list, u64));
        else if (type == Printer_Function_Type::_flt) return printer.printer_flt(file, va_arg(*arg_list, double));
        else if (type == Printer_Function_Type::_ptr) return printer.printer_ptr(file, va_arg(*arg_list, void*));
        else                                          return printer.printer_void(file);
    }
    return 0;

}

int maybe_fprintf(FILE* file, static_string format, int* pos, va_list* arg_list) {
    // %[flags][width][.precision][length]specifier
    // flags     ::= [+- #0]
    // width     ::= [<number>+ \*]
    // precision ::= \.[<number>+ \*]
    // length    ::= [h l L]
    // specifier ::= [c d i e E f g G o s u x X p n %]
    int end_pos = *pos;
    int written_len = 0;
    int used_arg_values = 1;

    // overstep flags:
    while(format[end_pos] == '+' ||
          format[end_pos] == '-' ||
          format[end_pos] == ' ' ||
          format[end_pos] == '#' ||
          format[end_pos] == '0')
        ++end_pos;

    // overstep width
    if (format[end_pos] == '*') {
        ++used_arg_values;
        ++end_pos;
    }
    else {
        while(format[end_pos] >= '0' && format[end_pos] <= '9')
            ++end_pos;
    }

    // overstep precision
    if (format[end_pos] == '.') {
        ++end_pos;
        if (format[end_pos] == '*') {
            ++used_arg_values;
            ++end_pos;
        }
        else {
            while(format[end_pos] >= '0' && format[end_pos] <= '9')
                ++end_pos;
        }
    }

    // overstep length:
    while(format[end_pos] == 'h' ||
          format[end_pos] == 'l' ||
          format[end_pos] == 'L')
        ++end_pos;

    //  check for actual built_in specifier
    if(format[end_pos] == 'c' ||
       format[end_pos] == 'd' ||
       format[end_pos] == 'i' ||
       format[end_pos] == 'e' ||
       format[end_pos] == 'E' ||
       format[end_pos] == 'f' ||
       format[end_pos] == 'g' ||
       format[end_pos] == 'G' ||
       format[end_pos] == 'o' ||
       format[end_pos] == 's' ||
       format[end_pos] == 'u' ||
       format[end_pos] == 'x' ||
       format[end_pos] == 'X' ||
       format[end_pos] == 'p' ||
       format[end_pos] == 'n' ||
       format[end_pos] == '%')
    {
        written_len = end_pos - *pos + 2;
        char* temp = (char*)alloca((written_len+1)* sizeof(char));
        temp[0] = '%';
        temp[1] = 0;
        strncpy(temp+1, format+*pos, written_len);
        temp[written_len] = 0;

        // printf("\ntest:: len(%s) = %d\n", temp, written_len+1);

        /// NOTE(Felix): Somehow we have to pass a copy of the list to vfprintf
        // because otherwise it destroys it on some platforms :(
        va_list arg_list_copy;
        va_copy(arg_list_copy, *arg_list);
        written_len = vfprintf(file, temp, arg_list_copy);
        va_end(arg_list_copy);

        // NOTE(Felix): manually overstep the args that vfprintf will have used
        //   all except the last used_arg will be integers (I hope) like for the
        //   padding and width and stuff, so we can overstep them with asking
        //   for a void*, but for the last one we need to check if it is a float
        //   so we can overstep it as a float.

        for (int i = 0; i < used_arg_values-1;  ++i) {
            va_arg(*arg_list, void*);
        }
        if (format[end_pos] == 'f' || format[end_pos] == 'g' || format[end_pos] == 'G' ||
            format[end_pos] == 'e' || format[end_pos] == 'E')
        {
            va_arg(*arg_list, f64);
        } else {
            va_arg(*arg_list, void*);
        }

        *pos = end_pos;
    }

    return written_len;
}


int print_va_args_to_file(FILE* file, static_string format, va_list* arg_list) {
    int printed_chars = 0;

    char c;
    int pos = -1;
    while ((c = format[++pos])) {
        if (c != '%') {
            putc(c, file);
            ++printed_chars;
        } else {
            c = format[++pos];
            int move = maybe_special_print(file, format, &pos, arg_list);
            if (move == 0) {
                move = maybe_fprintf(file, format, &pos, arg_list);
                if (move == -1) {
                    fputc('%', file);
                    fputc(c, file);
                    move = 1;
                }
            }
            printed_chars += move;
        }
    }

    return printed_chars;
}

int print_va_args_to_string(char** out, static_string format, va_list* arg_list) {
    FILE* t_file = tmpfile();
    if (!t_file) {
        return 0;
    }

    int num_printed_chars = print_va_args_to_file(t_file, format, arg_list);

    *out = (char*)malloc(sizeof(char) * (num_printed_chars+1));

    rewind(t_file);
    fread(*out, sizeof(char), num_printed_chars, t_file);
    (*out)[num_printed_chars] = '\0';

    fclose(t_file);

    return num_printed_chars;
}

int print_va_args(static_string format, va_list* arg_list) {
    return print_va_args_to_file(stdout, format, arg_list);
}

int print_to_string(char** out, static_string format, ...) {
    va_list arg_list;
    va_start(arg_list, format);

    FILE* t_file = tmpfile();
    if (!t_file) {
        return 0;
    }

    int num_printed_chars = print_va_args_to_file(t_file, format, &arg_list);
    va_end(arg_list);


    *out = (char*)malloc(sizeof(char) * (num_printed_chars+1));

    rewind(t_file);
    fread(*out, sizeof(char), num_printed_chars, t_file);
    (*out)[num_printed_chars] = '\0';

    fclose(t_file);

    return num_printed_chars;
}

int print_to_file(FILE* file, static_string format, ...) {
    va_list arg_list;
    va_start(arg_list, format);

    int num_printed_chars = print_va_args_to_file(file, format, &arg_list);

    va_end(arg_list);

    return num_printed_chars;
}

int print(static_string format, ...) {
    va_list arg_list;
    va_start(arg_list, format);

    int num_printed_chars = print_va_args_to_file(ftb_stdout, format, &arg_list);

    va_end(arg_list);

    return num_printed_chars;
}

int println(static_string format, ...) {
    va_list arg_list;
    va_start(arg_list, format);

    int num_printed_chars = print_va_args_to_file(ftb_stdout, format, &arg_list);
    num_printed_chars += print("\n");
    fflush(stdout);

    va_end(arg_list);

    return num_printed_chars;
}


int print_bool(FILE* f, u32 val) {
    return print_to_file(f, val ? "true" : "false");
}

int print_u32(FILE* f, u32 num) {
    return print_to_file(f, "%u", num);
}

int print_spaces(FILE* f, s32 num) {
    int sum = 0;

    while (num >= 8) {
        // println("%d", 8);
        sum += print_to_file(f, "        ");
        num -= 8;
    }
    while (num >= 4) {
        // println("%d", 4);
        sum += print_to_file(f, "    ");
        num -= 4;
    }
    while (num --> 0) {
        // println("%d", 1);
        sum += print_to_file(f, " ");
        num--;
    }
    return sum;
}


int print_u64(FILE* f, u64 num) {
    return print_to_file(f, "%llu", num);
}

int print_s32(FILE* f, s32 num) {
    return print_to_file(f, "%d", num);
}

int print_s64(FILE* f, s64 num) {
    return print_to_file(f, "%lld", num);
}

int print_flt(FILE* f, double arg) {
    return print_to_file(f, "%f", arg);
}

int print_str(FILE* f, char* str) {
    return print_to_file(f, "%s", str);
}

int print_color_start(FILE* f, char* str) {
    color_stack.append(str);
    return print_to_file(f, "%s", str);
}

int print_color_end(FILE* f) {
    --color_stack.count;
    if (color_stack.count == 0) {
        return print_to_file(f, "%s", console_normal);
    } else {
        return print_to_file(f, "%s", color_stack[color_stack.count-1]);
    }
}

int print_ptr(FILE* f, void* ptr) {
    if (ptr)
        return print_to_file(f, "%#0*X", sizeof(void*)*2+2, ptr);
    return print_to_file(f, "nullptr");
}

auto print_Str(FILE* f, String* str) -> s32 {
    return print_to_file(f, "%.*s", str->length, str->data);
}

auto print_str_line(FILE* f, char* str) -> s32 {
    u32 length = 0;
    while (str[length] != '\0') {
        if (str[length] == '\n')
            break;
        length++;
    }
    return print_to_file(f, "%.*s", length, str);
}

#ifdef FTB_USING_MATH
auto print_v2(FILE* f, V2* v2) -> s32 {
    return print_to_file(f, "{ %f %f }",
                         v2->x, v2->y);
}

auto print_v3(FILE* f, V3* v3) -> s32 {
    return print_to_file(f, "{ %f %f %f }",
                         v3->x, v3->y, v3->z);
}

auto print_v4(FILE* f, V4* v4) -> s32 {
    return print_to_file(f, "{ %f %f %f %f }",
                         v4->x, v4->y, v4->z, v4->w);
}

auto print_quat(FILE* f, Quat* quat) -> s32 {
    return print_v4(f, quat);
}

// NOTE(Felix): Matrices are in column major, but we print them in row major to
//   look more normal
auto print_m2x2(FILE* f, M2x2* m2x2) -> s32 {
    return print_to_file(f,
                         "{ %f %f\n"
                         "  %f %f }",
                         m2x2->_00, m2x2->_10,
                         m2x2->_01, m2x2->_11);
}

auto print_m3x3(FILE* f, M3x3* m3x3) -> s32 {
    return print_to_file(f,
                         "{ %f %f %f\n"
                         "  %f %f %f\n"
                         "  %f %f %f }",
                         m3x3->_00, m3x3->_10, m3x3->_20,
                         m3x3->_01, m3x3->_11, m3x3->_21,
                         m3x3->_02, m3x3->_12, m3x3->_22);
}

auto print_m4x4(FILE* f, M4x4* m4x4) -> s32 {
    return print_to_file(f,
                         "{ %f %f %f %f \n"
                         "  %f %f %f %f \n"
                         "  %f %f %f %f \n"
                         "  %f %f %f %f }",
                         m4x4->_00, m4x4->_10, m4x4->_20, m4x4->_30,
                         m4x4->_01, m4x4->_11, m4x4->_21, m4x4->_31,
                         m4x4->_02, m4x4->_12, m4x4->_22, m4x4->_32,
                         m4x4->_03, m4x4->_13, m4x4->_23, m4x4->_33);
}
#endif

void init_printer() {
#ifdef FTB_WINDOWS
    // enable colored terminal output for windows
    HANDLE hOut  = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING // g++ does not seem to define it
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
#endif
    color_stack.init();
    printer_map.init();
    type_map.init();

    register_printer("spaces",      print_spaces,      Printer_Function_Type::_32b);
    register_printer("u32",         print_u32,         Printer_Function_Type::_32b);
    register_printer("u64",         print_u64,         Printer_Function_Type::_64b);
    register_printer("bool",        print_bool,        Printer_Function_Type::_32b);
    register_printer("s64",         print_s64,         Printer_Function_Type::_64b);
    register_printer("s32",         print_s32,         Printer_Function_Type::_32b);
    register_printer("f32",         print_flt,         Printer_Function_Type::_flt);
    register_printer("f64",         print_flt,         Printer_Function_Type::_flt);
    register_printer("->char",      print_str,         Printer_Function_Type::_ptr);
    register_printer("->",          print_ptr,         Printer_Function_Type::_ptr);
    register_printer("color<",      print_color_start, Printer_Function_Type::_ptr);
    register_printer(">color",      print_color_end,   Printer_Function_Type::_void);
    register_printer("->Str",       print_Str,         Printer_Function_Type::_ptr);
    register_printer("->char_line", print_str_line,    Printer_Function_Type::_ptr);

#ifdef FTB_USING_MATH
    register_printer("->v2",   print_v2,   Printer_Function_Type::_ptr);
    register_printer("->v3",   print_v3,   Printer_Function_Type::_ptr);
    register_printer("->v4",   print_v4,   Printer_Function_Type::_ptr);
    register_printer("->quat", print_quat, Printer_Function_Type::_ptr);
    register_printer("->m2x2", print_m2x2, Printer_Function_Type::_ptr);
    register_printer("->m3x3", print_m3x3, Printer_Function_Type::_ptr);
    register_printer("->m4x4", print_m4x4, Printer_Function_Type::_ptr);
#endif
}

void deinit_printer() {
    color_stack.deinit();
    printer_map.deinit();
    type_map.deinit();
}

#ifndef FTB_NO_INIT_PRINTER
namespace {
    struct Printer_Initter {
        Printer_Initter() {
            init_printer();
        }
        ~Printer_Initter() {
            deinit_printer();
        }
    } p_initter;
}
#endif // FTB_NO_INIT_PRINTER
#endif // FTB_PRINT_IMPL
