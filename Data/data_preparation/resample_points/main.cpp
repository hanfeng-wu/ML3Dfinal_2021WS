#define _CRT_SECURE_NO_WARNINGS

#include <immintrin.h>
#include <random>
#include <mutex>
#include <math.h>
#include <omp.h>
#include <chrono>


#define FTB_PRINT_IMPL
#define FTB_HASHMAP_IMPL
#define FTB_STACKTRACE_IMPL
#define FTB_IO_IMPL
#define FTB_MATH_IMPL
#define FTB_MESH_IMPL
#include "./ftb/mesh.hpp"
#include "./ftb/print.hpp"

std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> dist_0_1(0.0f, 1.0f);
std::mutex rnd_lock;

const s32 NUM_THREADS = 20;

Array_List<f32>     probs_store[NUM_THREADS];
Array_List<f32> acc_probs_store[NUM_THREADS];
Array_List<u16>  face_nrs_store[NUM_THREADS];

const s32 obj_count = 2000;
char obj_file[obj_count][MAX_PATH] = {};
char gt_file[obj_count][MAX_PATH]  = {};


inline f32 rand_0_1() {
    return dist_0_1(e2);
}

u32 binary_search_prob(f32* probs, f32 needle, u32 count) {
    f32* base = probs;
    while (count > 1) {
        u32 middle = count / 2;
        base += (needle < base[middle]) ? 0 : middle;
        count -= middle;
    }

    return base-probs;
}

void resample_mesh(u32 mesh_nr, u32 thread_id, u32 num_samples) {
    const char* in_file_name  = obj_file[mesh_nr];
    const char* out_file_name = gt_file[mesh_nr];

    Mesh_Data m = load_obj(in_file_name, true);
    defer {
        m.vertices.deinit();
        m.faces.deinit();
    };

    FILE* out_file = fopen(out_file_name, "wb");
    if (!out_file) {
        fprintf(stderr, "outfile %s could not be opened", out_file_name);
        return;
    }
    defer {
        fclose(out_file);
    };

    probs_store[thread_id].reserve(m.faces.count);
    acc_probs_store[thread_id].reserve(m.faces.count);
    face_nrs_store[thread_id].reserve(m.faces.count);

    f32* probs     = probs_store[thread_id].data;
    f32* acc_probs = acc_probs_store[thread_id].data;
    u16* face_nrs  = face_nrs_store[thread_id].data;

    // number the face_nr array
    for (u32 f_idx = 0; f_idx < m.faces.count; ++f_idx) {
        face_nrs[f_idx] = f_idx;
    }

    // calcualte sizes
    for (u32 f_idx = 0; f_idx < m.faces.count; ++f_idx) {
        V3 v1 = m.vertices[m.faces[f_idx].v1].position;
        V3 v2 = m.vertices[m.faces[f_idx].v2].position;
        V3 v3 = m.vertices[m.faces[f_idx].v3].position;

        V3 s1 = v2 - v1;
        V3 s2 = v3 - v1;

        V3 crss = cross(s1, s2);
        probs[f_idx] = dot(crss, crss);
    }

    // NOTE(Felix): probs now contain the squared sizes of the faces, we
    //   have to sqrt all of them, and use simd for that for speed

    const u8 simd_size = 8;
    u32 i;
    f32 area_sum = 0.0f;

    // surface areas
    {
        __m256 simd_area_sum = _mm256_set1_ps(0.0f);

        for (i = 0; i + simd_size <= m.faces.count; i += simd_size) {
            __m256 squares = _mm256_loadu_ps(probs+i);
            __m256 areas = _mm256_sqrt_ps(squares);
            simd_area_sum = _mm256_add_ps(simd_area_sum, areas);
            _mm256_storeu_ps(probs+i, areas);
        }

        // reminder loop
        for (; i < m.faces.count; ++i) {
            probs[i] = sqrtf(probs[i]);
            area_sum += probs[i];
        }

        // NOTE(Felix): calculate total surface area (actually 2* surface area,
        //   since with the cross product we always calculate the double of the sice
        //   of the triangle but it is okay since we normalize the faces so linear
        //   factors don't matter)
        for (int j = 0; j < simd_size; ++j) {
            area_sum += ((f32*)(&simd_area_sum))[j];
        }

    }

    // normalization
    {
        f32 prob_factor = 1.0f / area_sum;
        __m256 simd_prob_factor = _mm256_set1_ps(prob_factor);

        for (i = 0; i + simd_size <= m.faces.count; i += simd_size) {
            __m256 simd_areas = _mm256_loadu_ps(probs+i);
            __m256 simd_probs = _mm256_mul_ps(simd_areas, simd_prob_factor);
            _mm256_storeu_ps(probs+i, simd_probs);
        }

        // remainder loop
        for (; i < m.faces.count; ++i) {
            probs[i] *= prob_factor;
        }
    }

    {
        acc_probs[0] = 0;
        // Fill out the accumulated probs
        for (u32 i = 1; i < m.faces.count; ++i) {
            acc_probs[i] = acc_probs[i-1] + probs[i-1];
        }
    }

    for (u32 s = 0; s < num_samples; ++s) {
        rnd_lock.lock();
        f32 face_idx_rng = rand_0_1();
        f32 r1 = rand_0_1();
        f32 r2 = rand_0_1();
        rnd_lock.unlock();

        u32 prob_idx = binary_search_prob(acc_probs, face_idx_rng, m.faces.count);
        u32 face_idx = face_nrs[prob_idx];

        V3 v1 = m.vertices[m.faces[face_idx].v1].position;
        V3 v2 = m.vertices[m.faces[face_idx].v2].position;
        V3 v3 = m.vertices[m.faces[face_idx].v3].position;

        f32 sqrt_r1 = sqrtf(r1);

        f32 u = 1-sqrt_r1;
        f32 v = sqrt_r1*(1-r2);
        f32 w = sqrt_r1*r2;

        V3 new_point = u*v1 + v*v2 + w*v3;

        fprintf(out_file, "v %.17f %.17f %.17f\n", new_point.x, new_point.y, new_point.z);
    }
}

int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    defer {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        println("\nElapsed time: %.2fs",
                0.001f * std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count());
    };

    init_printer();

    println("Running python script...");
    if (system("python obj_lister.py") != 0) {
        fprintf(stderr, "could not run python script\n");
        return 1;
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        probs_store[i].init(500000);
        acc_probs_store[i].init(500000);
        face_nrs_store[i].init(500000);
    }

    FILE* file = fopen("./mesh_dirs.txt", "r");
    if (!file) {
        fprintf(stderr, "could not open mesh list file\n");
        return 1;
    }
    defer {
        fclose(file);
    };

    for (int i = 0; i < obj_count; ++i) {
        fscanf(file, "%s %s\n", obj_file[i], gt_file[i]);
    }


    std::mutex print_lock;

#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
    for (int i = 0; i < obj_count; ++i) {
        print_lock.lock();
        {
            println("%d\t%d", omp_get_thread_num(), i);
        }
        print_lock.unlock();
        resample_mesh(i, omp_get_thread_num(), 4000);
    }

    println("all done.");

    return 0;
}
