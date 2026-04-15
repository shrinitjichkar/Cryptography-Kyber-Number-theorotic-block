#include <stddef.h>
#include <stdint.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#include "invntt_optimized.h"
#include "kyber_params.h"
#include "kyber_zetas.h"

/*
Paper-based reference style (kept here as requested):

void invntt_ref(int16_t r[256]) {
    unsigned int start, len, j, k;
    int16_t t, zeta;
    const int16_t f = 1441; // mont^2/128
    k = 127;
    for(len = 2; len <= 128; len <<= 1) {
        for(start = 0; start < 256; start = j + len) {
            zeta = zetas[k--];
            for(j = start; j < start + len; j++) {
                t = r[j];
                r[j] = barrett_reduce(t + r[j + len]);
                r[j + len] = r[j + len] - t;
                r[j + len] = fqmul(zeta, r[j + len]);
            }
        }
    }
    for(j = 0; j < 256; j++) {
        r[j] = fqmul(r[j], f);
    }
}
*/

static inline int16_t montgomery_reduce(int32_t a) {
    int16_t t = (int16_t)a * KYBER_QINV;
    t = (a - (int32_t)t * KYBER_Q) >> 16;
    return t;
}

static inline int16_t barrett_reduce(int16_t a) {
    const int16_t v = (int16_t)((1U << 26) / KYBER_Q + 1U);
    int16_t t = (int16_t)(((int32_t)v * a + (1 << 25)) >> 26);
    return a - t * KYBER_Q;
}

static inline int16_t fqmul(int16_t a, int16_t b) {
    return montgomery_reduce((int32_t)a * b);
}

void invntt_optimized(int16_t r[KYBER_N]) {
    /* IMPROVEMENT-1: LUT access is block-reused (one zeta per start block). */
    /* IMPROVEMENT-2: SIMD batched load/store for butterfly pairs in hot loops. */
    /* IMPROVEMENT-3: restrict-style non-aliasing local pointers aid scheduling. */
    int16_t *restrict out = r;
    size_t k = KYBER_ZETAS_LEN - 1; /* equivalent to paper's k=127 */

    for (size_t len = 2; len <= KYBER_N / 2; len <<= 1) {
        for (size_t start = 0; start < KYBER_N; start += 2 * len) {
            const int16_t zeta = kyber_zetas[k--];

#ifdef __wasm_simd128__
            size_t j = start;
            for (; j + 8 <= start + len; j += 8) {
                int16_t lo_buf[8];
                int16_t hi_buf[8];
                wasm_v128_store(lo_buf, wasm_v128_load(&out[j]));
                wasm_v128_store(hi_buf, wasm_v128_load(&out[j + len]));

                /* vector add/sub first, then modular operations per lane */
                v128_t vlo = wasm_v128_load(lo_buf);
                v128_t vhi = wasm_v128_load(hi_buf);
                v128_t vsum = wasm_i16x8_add(vlo, vhi);
                v128_t vdiff = wasm_i16x8_sub(vhi, vlo);
                wasm_v128_store(lo_buf, vsum);
                wasm_v128_store(hi_buf, vdiff);

                for (int lane = 0; lane < 8; ++lane) {
                    lo_buf[lane] = barrett_reduce(lo_buf[lane]);
                    hi_buf[lane] = fqmul(zeta, hi_buf[lane]);
                }

                wasm_v128_store(&out[j], wasm_v128_load(lo_buf));
                wasm_v128_store(&out[j + len], wasm_v128_load(hi_buf));
            }

            for (; j < start + len; ++j) {
                int16_t t = out[j];
                out[j] = barrett_reduce((int16_t)(t + out[j + len]));
                out[j + len] = fqmul(zeta, (int16_t)(out[j + len] - t));
            }
#else
            for (size_t j = start; j < start + len; ++j) {
                int16_t t = out[j];
                out[j] = barrett_reduce((int16_t)(t + out[j + len]));
                out[j + len] = fqmul(zeta, (int16_t)(out[j + len] - t));
            }
#endif
        }
    }

    {
        const int16_t f = 1441; /* paper constant: mont^2/128 */
        for (size_t j = 0; j < KYBER_N; ++j) {
            out[j] = fqmul(out[j], f);
        }
    }
}

