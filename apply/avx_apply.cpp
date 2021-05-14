#include <immintrin.h>

#include "avx_apply.h"


template<size_t WordLen>
struct RegI {
    typedef __m128i Type;
    const static size_t FetchSize = 2;
};

template<>
struct RegI<256> {
    typedef __m256i Type;
    const static size_t FetchSize = 4;
};

template<>
struct RegI<512> {
    typedef __m512i Type;
    const static size_t FetchSize = 8;
};

template<size_t WordLen>
using TRegI = typename RegI<WordLen>::Type;

template<size_t WordLen>
const size_t FetchSize = RegI<WordLen>::FetchSize;

template<size_t WordLen>
static void NFORCED_INLINE DoFetchWord(const ui8* value, const int *d0, TRegI<WordLen> &val) {
    if constexpr (WordLen == 128) {
        ui16 vals = *(ui16*)value;
        val = _mm_add_epi64(val, _mm_setr_epi64(_mm_cvtsi32_si64(d0[vals & 0xff]),
                                                _mm_cvtsi32_si64(d0[(vals >> (8 * 1)) & 0xff])));
    } else if constexpr (WordLen == 256) {
        ui32 vals = *(ui32*)value;
        val = _mm256_add_epi64(val, _mm256_setr_epi64x((ui32)d0[vals & 0xff],
                                                       (ui32)d0[(vals >> (8 * 1)) & 0xff],
                                                       (ui32)d0[(vals >> (8 * 2)) & 0xff],
                                                       (ui32)d0[(vals >> (8 * 3)) & 0xff]));
    } else if constexpr (WordLen == 512) {
        ui64 vals = *(ui64*)value;
        val = _mm512_add_epi64(val, _mm512_setr_epi64((ui32)d0[vals & 0xff],
                                                      (ui32)d0[(vals >> (8 * 1)) & 0xff],
                                                      (ui32)d0[(vals >> (8 * 2)) & 0xff],
                                                      (ui32)d0[(vals >> (8 * 3)) & 0xff],
                                                      (ui32)d0[(vals >> (8 * 4)) & 0xff],
                                                      (ui32)d0[(vals >> (8 * 5)) & 0xff],
                                                      (ui32)d0[(vals >> (8 * 6)) & 0xff],
                                                      (ui32)d0[(vals >> (8 * 7)) & 0xff]));
    }
}

template<size_t WordLen>
static NFORCED_INLINE void DoFetchMany(TRegI<WordLen>* datas, const int *d0, const int *d1, const int *d2, const int *d3, const ui8 bins[4][WordLen], size_t& docIndex) {
    NFORCE_UNROLL
    for (size_t i = 0; i < 8; ++i) {
        TRegI<WordLen> val = datas[docIndex];
        DoFetchWord<WordLen>(&bins[0][FetchSize<WordLen> * docIndex], d0, val);
        DoFetchWord<WordLen>(&bins[1][FetchSize<WordLen> * docIndex], d1, val);
        DoFetchWord<WordLen>(&bins[2][FetchSize<WordLen> * docIndex], d2, val);
        DoFetchWord<WordLen>(&bins[3][FetchSize<WordLen> * docIndex], d3, val);
        datas[docIndex] = val;
        ++docIndex;
    }
}

template <size_t X, size_t WordLen>
static void NINLINE_UNLESS_PROFILE DoFetchAllFromFourTrees(const int *d0, const int *d1, const int *d2, const int *d3, const ui8 bins[4][WordLen], ui64 (*datasOrig)[WordLen], size_t num) {
    TRegI<WordLen>* datas = (TRegI<WordLen>*) datasOrig;
    size_t docIndex = 0;
    NFORCE_UNROLL
    for (size_t i = 1; i < X; ++i) {
        DoFetchMany<WordLen>(datas, d0, d1, d2, d3, bins, docIndex);
    }

    if (num - FetchSize<WordLen> * docIndex == WordLen / 8) {
        DoFetchMany<WordLen>(datas, d0, d1, d2, d3, bins, docIndex);
    } else {
        while (FetchSize<WordLen> * docIndex < num) {
            TRegI<WordLen> val = datas[docIndex];
            DoFetchWord<WordLen>(&bins[0][FetchSize<WordLen> * docIndex], d0, val);
            DoFetchWord<WordLen>(&bins[1][FetchSize<WordLen> * docIndex], d1, val);
            DoFetchWord<WordLen>(&bins[2][FetchSize<WordLen> * docIndex], d2, val);
            DoFetchWord<WordLen>(&bins[3][FetchSize<WordLen> * docIndex], d3, val);
            datas[docIndex] = val;
            ++docIndex;
        }
    }
}

template <size_t X, size_t WordLen>
static void NINLINE_UNLESS_PROFILE DoFetchAllFromOneTree(const int *d0, const ui8 bins[WordLen], ui64 (*datasOrig)[WordLen], size_t num) {
    TRegI<WordLen>* datas = (TRegI<WordLen>*) datasOrig;
    size_t docIndex = 0;
    NFORCE_UNROLL
    for (size_t i = 1; i < X; ++i) {
        NFORCE_UNROLL
        for (size_t i = 0; i < 8; ++i) {
            TRegI<WordLen> val = datas[docIndex];
            DoFetchWord<WordLen>(&bins[FetchSize<WordLen> * docIndex], d0, val);
            datas[docIndex] = val;
            ++docIndex;
        }
    }

    while (FetchSize<WordLen> * docIndex < num) {
        TRegI<WordLen> val = datas[docIndex];
        DoFetchWord<WordLen>(&bins[FetchSize<WordLen> * docIndex], d0, val);
        datas[docIndex] = val;
        ++docIndex;
    }
}

int zeroes[1 << 8];

template <size_t WordLen>
struct TFetcher {
    using TLeaf = int;
    alignas(0x40) ui8 Bins[4][WordLen]; // leaves indexes
    alignas(0x40) const TLeaf *Data[4]; // trees data pointers
    alignas(0x40) ui64 Vals[WordLen]; // trees leaves sum
    size_t Num = 0;
    size_t Count = 0;

    TFetcher(size_t num)
            : Num(num)
    {
#if defined(_msan_enabled_)
        // Bypass use-of-uninitialized-value error
        memset(Bins, 0, sizeof(Bins));
        memset(Data, 0, sizeof(Data));
#endif
        memset(Vals, 0, sizeof(Vals));
    }

    template <size_t X>
    NFORCED_INLINE ui8 *Alloc(const TLeaf *data) {
        DoFetch<X>();
        ui8 *res = Bins[Count];
        Data[Count] = data;
        if constexpr (X > 1) {
            Y_PREFETCH_READ(data + 0 * 16, 3);
            Y_PREFETCH_READ(data + 1 * 16, 3);
            Y_PREFETCH_READ(data + 2 * 16, 3);
            Y_PREFETCH_READ(data + 3 * 16, 3);
        }
        ++Count;
        return res;
    }

    template <size_t X>
    NFORCED_INLINE void DoFlush() {
        if (DoFetch<X>())
            return;
        for (size_t i = Count; i < 4; ++i) {
            Alloc<X>(zeroes);
        }
        DoFetch<X>();
    }

    template <size_t X>
    NFORCED_INLINE bool DoFetch() {
        if (Count == 4){
            DoFetchAllFromFourTrees<X, WordLen>(Data[0], Data[1], Data[2], Data[3], Bins, &Vals, Num);
            Count = 0;
        }
        return Count == 0;
    }
};

////////////////////////////////////////////////////////////

enum TFetcherType {
    FT_FetchFourTrees,
    FT_FetchOneTree,
    FT_GaitherFetch,
    FT_ShuffleFetch
};

template<TFetcherType FetcherType, size_t WordLen>
struct TResType {
    typedef ui8* Type;
};

template<size_t WordLen>
struct TResType<FT_GaitherFetch, WordLen> {
    typedef TFetcher<WordLen>* Type;
};

template<size_t WordLen>
struct TResType<FT_ShuffleFetch, WordLen> {
    typedef TFetcher<WordLen>* Type;
};

template<size_t Num, size_t Depth, TFetcherType FetcherType = FT_FetchFourTrees>
static void NINLINE_UNLESS_PROFILE DoMXLane_128(const ui32 *val, const ui32 *indices, const ui32 *end, typename TResType<FetcherType, 128>::Type res) {
    const ui32 *fetch = Min(end - Depth, indices + 10);
    __m128i v[Num];

    if constexpr (Depth == 0) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            v[i] = _mm_setzero_si128();
        }
    }

    NFORCE_UNROLL
    for (size_t i = 0; i < Depth; ++i) {
        Y_PREFETCH_READ(val + fetch[i], 3);
        __m128i value = _mm_load_si128((__m128i *)(val + indices[i]));
        __m128i positions = _mm_slli_epi16(_mm_set1_epi8(0x01), i);
        NFORCE_UNROLL
        for (size_t j = 0; j < Num; ++j) {
            if (i == 0) {
                if (j == 0) {
                    v[j] = _mm_and_si128(value, positions);
                } else {
                    v[j] = _mm_and_si128(_mm_srli_epi16(value, j), positions);
                }
            } else {
                if (i < j) {
                    v[j] = _mm_add_epi32(v[j], _mm_and_si128(_mm_srli_epi16(value, j - i), positions));
                } else if (i == j) {
                    v[j] = _mm_add_epi32(v[j], _mm_and_si128(value, positions));
                } else if (i >= j) {
                    v[j] = _mm_add_epi32(v[j], _mm_and_si128(_mm_slli_epi16(value, i - j), positions));
                }
            }
        }
    }

    Y_PREFETCH_READ(indices, 3);
    if constexpr (FetcherType != FT_GaitherFetch) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            _mm_store_si128((__m128i *)(res + 16 * i), v[i]);
        }
    } else if constexpr (FetcherType == FT_GaitherFetch) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            NFORCE_UNROLL
            for (size_t j = 0; j < 4; ++j) {
                __m128i vp;
                if (j == 0) {
                    vp = v[i];
                } else if (j == 1) {
                    vp = _mm_srli_si128(v[i], 1 * 4);
                } else if (j == 2) {
                    vp = _mm_srli_si128(v[i], 2 * 4);
                } else if (j == 3) {
                    vp = _mm_srli_si128(v[i], 3 * 4);
                } else {
                    Y_FAIL("Wrong Iterations Number");
                }
                __m128i data = _mm_i32gather_epi32(res->Data[0], _mm_cvtepu8_epi32(vp), sizeof(ui32));
                _mm_store_si128(   (__m128i *)(res->Vals + 4 * (j + 4 * i)), _mm_add_epi64(
                        _mm_load_si128((__m128i *)(res->Vals + 4 * (j + 4 * i))), _mm_cvtepu32_epi64(data)));
                _mm_store_si128(   (__m128i *)(res->Vals + 4 * (j + 4 * i) + 2), _mm_add_epi64(
                        _mm_load_si128((__m128i *)(res->Vals + 4 * (j + 4 * i) + 2)), _mm_cvtepu32_epi64(_mm_srli_si128(data, 8))));
            }
        }
    }
}
template<size_t Num, size_t Depth, TFetcherType FetcherType = FT_FetchFourTrees>
static void NINLINE_UNLESS_PROFILE DoMXLane_256(const ui32 *val, const ui32 *indices, const ui32 *end, typename TResType<FetcherType, 256>::Type res) {
    const ui32 *fetch = Min(end - Depth, indices + 10);
    __m256i v[Num];
    if constexpr (Depth == 0) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            v[i] = _mm256_setzero_si256();
        }
    }

    NFORCE_UNROLL
    for (size_t i = 0; i < Depth; ++i) {
        Y_PREFETCH_READ(val + fetch[i], 3);
        __m256i value = _mm256_load_si256((__m256i *)(val + indices[i] * 2));
        __m256i positions = _mm256_slli_epi16(_mm256_set1_epi8(0x01), i);
        NFORCE_UNROLL
        for (size_t j = 0; j < Num; ++j) {
            if (i == 0) {
                if (j == 0) {
                    v[j] = _mm256_and_si256(value, positions);
                } else {
                    v[j] = _mm256_and_si256(_mm256_srli_epi16(value, j), positions);
                }
            } else {
                if (i < j) {
                    v[j] = _mm256_add_epi32(v[j], _mm256_and_si256(_mm256_srli_epi16(value, j - i), positions));
                } else if (i == j) {
                    v[j] = _mm256_add_epi32(v[j], _mm256_and_si256(value, positions));
                } else if (i >= j) {
                    v[j] = _mm256_add_epi32(v[j], _mm256_and_si256(_mm256_slli_epi16(value, i - j), positions));
                }
            }
        }
    }

    Y_PREFETCH_READ(indices, 3);
    if constexpr (FetcherType != FT_GaitherFetch && FetcherType != FT_ShuffleFetch) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            _mm256_store_si256((__m256i *)(res + 32 * i), v[i]);
        }
    } else {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            NFORCE_UNROLL
            for (size_t j = 0; j < 4; ++j) {
                __m256i vp;
                if (j == 0) {
                    vp = _mm256_unpacklo_epi16(_mm256_unpacklo_epi8(v[i], _mm256_setzero_si256()), _mm256_setzero_si256());
                } else if (j == 1) {
                    vp = _mm256_unpackhi_epi16(_mm256_unpacklo_epi8(v[i], _mm256_setzero_si256()), _mm256_setzero_si256());
                } else if (j == 2) {
                    vp = _mm256_unpacklo_epi16(_mm256_unpackhi_epi8(v[i], _mm256_setzero_si256()), _mm256_setzero_si256());
                } else if (j == 3) {
                    vp = _mm256_unpackhi_epi16(_mm256_unpackhi_epi8(v[i], _mm256_setzero_si256()), _mm256_setzero_si256());
                } else {
                    Y_FAIL("Wrong Iterations Number");
                }
                __m256i data = _mm256_i32gather_epi32(res->Data[0], vp, sizeof(ui32));
                _mm256_store_si256(   (__m256i *)(res->Vals + 8 * (j + 4 * i)), _mm256_add_epi64(
                        _mm256_load_si256((__m256i *)(res->Vals + 8 * (j + 4 * i))), _mm256_cvtepu32_epi64(_mm256_castsi256_si128(data))));
                _mm256_store_si256(   (__m256i *)(res->Vals + 8 * (j + 4 * i) + 4), _mm256_add_epi64(
                        _mm256_load_si256((__m256i *)(res->Vals + 8 * (j + 4 * i) + 4)), _mm256_cvtepu32_epi64(_mm256_extractf128_si256(data, 1))));
            }
        }
    }
}

template<size_t Num, size_t Depth, TFetcherType FetcherType = FT_FetchFourTrees>
static void NINLINE_UNLESS_PROFILE DoMXLane_512(const ui32 *val, const ui32 *indices, const ui32 *end, typename TResType<FetcherType, 512>::Type res) {
    const ui32 *fetch = Min(end - Depth, indices + 10);
    __m512i v[Num];
    if constexpr (Depth == 0) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            v[i] = _mm512_setzero_si512();
        }
    }

    NFORCE_UNROLL
    for (size_t i = 0; i < Depth; ++i) {
        Y_PREFETCH_READ(val + fetch[i], 3);
        __m512i value = _mm512_load_si512((__m512i *)(val + indices[i] * 4));
        __m512i positions = _mm512_slli_epi16(_mm512_set1_epi8(0x01), i);
        NFORCE_UNROLL
        for (size_t j = 0; j < Num; ++j) {
            if (i == 0) {
                if (j == 0) {
                    v[j] = _mm512_and_si512(value, positions);
                } else {
                    v[j] = _mm512_and_si512(_mm512_srli_epi16(value, j), positions);
                }
            } else {
                if (i < j) {
                    v[j] = _mm512_add_epi32(v[j], _mm512_and_si512(_mm512_srli_epi16(value, j - i), positions));
                } else if (i == j) {
                    v[j] = _mm512_add_epi32(v[j], _mm512_and_si512(value, positions));
                } else if (i >= j) {
                    v[j] = _mm512_add_epi32(v[j], _mm512_and_si512(_mm512_slli_epi16(value, i - j), positions));
                }
            }
        }
    }

    Y_PREFETCH_READ(indices, 3);
    if constexpr (FetcherType != FT_GaitherFetch && FetcherType != FT_ShuffleFetch) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            _mm512_store_si512(res + 64 * i, v[i]);
        }
    } else if constexpr (FetcherType == FT_GaitherFetch) {
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            NFORCE_UNROLL
            for (size_t j = 0; j < 4; ++j) {
                __m512i vp;
                if (j == 0) {
                    vp = _mm512_unpacklo_epi16(_mm512_unpacklo_epi8(v[i], _mm512_setzero_si512()), _mm512_setzero_si512());
                } else if (j == 1) {
                    vp = _mm512_unpackhi_epi16(_mm512_unpacklo_epi8(v[i], _mm512_setzero_si512()), _mm512_setzero_si512());
                } else if (j == 2) {
                    vp = _mm512_unpacklo_epi16(_mm512_unpackhi_epi8(v[i], _mm512_setzero_si512()), _mm512_setzero_si512());
                } else if (j == 3) {
                    vp = _mm512_unpackhi_epi16(_mm512_unpackhi_epi8(v[i], _mm512_setzero_si512()), _mm512_setzero_si512());
                } else {
                    Y_FAIL("Wrong Iterations Number");
                }
                __m512i data = _mm512_i32gather_epi32(vp, res->Data[0], sizeof(ui32));
                _mm512_store_si512(   res->Vals + 16 * (j + 4 * i), _mm512_add_epi64(
                        _mm512_load_si512(res->Vals + 16 * (j + 4 * i)), _mm512_cvtepu32_epi64(_mm512_castsi512_si256(data))));
                _mm512_store_si512(   res->Vals + 16 * (j + 4 * i) + 8, _mm512_add_epi64(
                        _mm512_load_si512(res->Vals + 16 * (j + 4 * i) + 8), _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(data, 1))));
            }
        }
    } else {
        static_assert(Depth >= 4);
        constexpr size_t RegNum = (1 << Depth) * 32 / 512;
        __m512i arr[RegNum];
        NFORCE_UNROLL
        for (size_t i = 0; i < RegNum; ++i) {
            arr[i] = _mm512_load_si512((__m512i *)(res->Data[0] + i * 16));
        }
        NFORCE_UNROLL
        for (size_t i = 0; i < Num; ++i) {
            NFORCE_UNROLL
            for (size_t j = 0; j < 4; ++j) {
                __m128i vp;
                if        (j == 0) {
                    vp = _mm512_extracti32x4_epi32(v[i], 0);
                } else if (j == 1) {
                    vp = _mm512_extracti32x4_epi32(v[i], 1);
                } else if (j == 2) {
                    vp = _mm512_extracti32x4_epi32(v[i], 2);
                } else {
                    vp = _mm512_extracti32x4_epi32(v[i], 3);
                }
                __m512i shuffle = _mm512_cvtepu8_epi32(_mm_and_si128(vp, _mm_set1_epi8(0x0f)));
                __m128i reg_index = _mm_srli_epi64(_mm_and_si128(vp, _mm_set1_epi8(0xf0)), 4);
                __m512i data = _mm512_setzero_si512();
                NFORCE_UNROLL
                for (size_t k = 0; k < RegNum; ++k) {
                    data = _mm512_or_si512(data, _mm512_maskz_permutexvar_epi32(
                            _mm_cmpeq_epu8_mask(reg_index, _mm_set1_epi8(k)),
                            shuffle,
                            arr[k]
                    ));
                }
                _mm512_store_si512(   res->Vals + 16 * (j + 4 * i), _mm512_add_epi64(
                        _mm512_load_si512(res->Vals + 16 * (j + 4 * i)), _mm512_cvtepu32_epi64(_mm512_castsi512_si256(data))));
                _mm512_store_si512(   res->Vals + 16 * (j + 4 * i) + 8, _mm512_add_epi64(
                        _mm512_load_si512(res->Vals + 16 * (j + 4 * i) + 8), _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(data, 1))));
            }
        }
    }
}

template<size_t Num, size_t Depth>
static void NINLINE_UNLESS_PROFILE DoMXLaneOrdered_128(const ui32 *val, const ui32 *indices, const ui32 *end, ui8 *res) {
    const ui32 *fetch = Min(end - Depth, indices + 10);

    __m128i v[Num];
    NFORCE_UNROLL
    for (size_t i = 0; i < Num; ++i) {
        v[i] = _mm_setzero_si128();
    }

    NFORCE_UNROLL
    for (size_t i = 0; i < Depth; ++i) {
        Y_PREFETCH_READ(val + fetch[i], 3);
        const ui16 *b = reinterpret_cast<const ui16*>(val + indices[i]);
        NFORCE_UNROLL
        for (size_t j = 0; j < Num; ++j) {
            v[j] = _mm_or_epi32(v[j], _mm_slli_epi64(_mm_maskz_set1_epi8(b[j], 1), i));
        }
    }

    Y_PREFETCH_READ(indices, 3);
    NFORCE_UNROLL
    for (size_t i = 0; i < Num; ++i) {
        _mm_store_si128((__m128i *)(res + 16 * i), v[i]);
    }
}

template<size_t Num, size_t Depth>
static void NINLINE_UNLESS_PROFILE DoMXLaneOrdered_256(const ui32 *val, const ui32 *indices, const ui32 *end, ui8 *res) {
    const ui32 *fetch = Min(end - Depth, indices + 10);

    __m256i v[Num];
    NFORCE_UNROLL
    for (size_t i = 0; i < Num; ++i) {
        v[i] = _mm256_setzero_si256();
    }

    NFORCE_UNROLL
    for (size_t i = 0; i < Depth; ++i) {
        Y_PREFETCH_READ(val + fetch[i], 3);
        const ui32 *b = reinterpret_cast<const ui32*>(val + indices[i] * 2);
        NFORCE_UNROLL
        for (size_t j = 0; j < Num; ++j) {
            v[j] = _mm256_or_si256(v[j], _mm256_slli_epi16(_mm256_maskz_set1_epi8(b[j], 1), i));
        }
    }

    Y_PREFETCH_READ(indices, 3);
    NFORCE_UNROLL
    for (size_t i = 0; i < Num; ++i) {
        _mm256_store_si256((__m256i *)(res + 32 * i), v[i]);
    }
}

template<size_t Num, size_t Depth>
static void NINLINE_UNLESS_PROFILE DoMXLaneOrdered_512(const ui32 *val, const ui32 *indices, const ui32 *end, ui8 *res) {
    const ui32 *fetch = Min(end - Depth, indices + 10);

    __m512i v[Num];
    NFORCE_UNROLL
    for (size_t i = 0; i < Num; ++i) {
        v[i] = _mm512_setzero_si512();
    }

    NFORCE_UNROLL
    for (size_t i = 0; i < Depth; ++i) {
        Y_PREFETCH_READ(val + fetch[i], 3);
        const ui64 *b = reinterpret_cast<const ui64*>(val + indices[i] * 4);
        NFORCE_UNROLL
        for (size_t j = 0; j < Num; ++j) {
            v[j] = _mm512_or_si512(v[j], _mm512_slli_epi16(_mm512_maskz_set1_epi8(b[j], 1), i));
        }
    }

    Y_PREFETCH_READ(indices, 3);
    NFORCE_UNROLL
    for (size_t i = 0; i < Num; ++i) {
        _mm512_store_si512((__m512i *)(res + 64 * i), v[i]);
    }
}

template<size_t Num, size_t WordLen, bool Ordered, TFetcherType FetcherType, size_t Depth>
static void CalculateAllTreesOnDepth(
        const NMatrixnet::TMnSseStaticMeta& info,
        const typename TFetcher<WordLen>::TLeaf*& data,
        const ui32*& beginDataIndices,
        const ui32* endDataIndices,
        TFetcher<WordLen>* fetcher,
        const ui32* val,
        int& proceed
) {
    const int size = info.GetSizeToCount(Depth);
    constexpr size_t MinShuffleDepth = (WordLen == 512 ? 4 : 3);
    constexpr bool FetchOneTree = (FetcherType == FT_FetchOneTree || FetcherType == FT_ShuffleFetch && Depth < MinShuffleDepth);
    for (int i = 0; i < size; ++i) {
        if constexpr (FetcherType == FT_GaitherFetch || FetcherType == FT_ShuffleFetch && Depth >= MinShuffleDepth) {
            fetcher->Data[0] = data;
            if        constexpr (WordLen == 128) {
                DoMXLane_128<Num, Depth, FetcherType>(val, beginDataIndices, endDataIndices, fetcher);
            } else if constexpr (WordLen == 256) {
                DoMXLane_256<Num, Depth, FetcherType>(val, beginDataIndices, endDataIndices, fetcher);
            } else if constexpr (WordLen == 512) {
                DoMXLane_512<Num, Depth, FetcherType>(val, beginDataIndices, endDataIndices, fetcher);
            } else {
                Y_FAIL("Unsupported wordlen");
            }
        } else {
            if constexpr (Ordered) {
                if        constexpr (WordLen == 128) {
                    DoMXLaneOrdered_128<Num, Depth>(val, beginDataIndices, endDataIndices,
                                                    FetchOneTree ? fetcher->Bins[0] : fetcher->template Alloc<Num>(data));
                } else if constexpr (WordLen == 256) {
                    DoMXLaneOrdered_256<Num, Depth>(val, beginDataIndices, endDataIndices,
                                                    FetchOneTree ? fetcher->Bins[0] : fetcher->template Alloc<Num>(data));
                } else if constexpr (WordLen == 512) {
                    DoMXLaneOrdered_512<Num, Depth>(val, beginDataIndices, endDataIndices,
                                                    FetchOneTree ? fetcher->Bins[0] : fetcher->template Alloc<Num>(data));
                } else {
                    Y_FAIL("Unsupported wordlen");
                }
            } else {
                if        constexpr (WordLen == 128) {
                    DoMXLane_128<Num, Depth>(val, beginDataIndices, endDataIndices,
                                             FetchOneTree ? fetcher->Bins[0] : fetcher->template Alloc<Num>(data));
                } else if constexpr (WordLen == 256) {
                    DoMXLane_256<Num, Depth>(val, beginDataIndices, endDataIndices,
                                             FetchOneTree ? fetcher->Bins[0] : fetcher->template Alloc<Num>(data));
                } else if constexpr (WordLen == 512) {
                    DoMXLane_512<Num, Depth>(val, beginDataIndices, endDataIndices,
                                             FetchOneTree ? fetcher->Bins[0] : fetcher->template Alloc<Num>(data));
                } else {
                    Y_FAIL("Unsupported wordlen");
                }
            }
            if constexpr (FetchOneTree) {
                DoFetchAllFromOneTree<Num, WordLen>(data, fetcher->Bins[0], &fetcher->Vals, fetcher->Num);
            }
        }
        beginDataIndices += Depth;
        if constexpr (FetcherType == FT_ShuffleFetch && Depth < 4) {
            data += 512 / 32;
        } else {
            data += (1 << Depth);
        }
    }
    proceed += size;
}

////////////////////////////////////////////////////////////

template <size_t Num, size_t WordLen, bool Ordered, TFetcherType FetcherType>
static void CalculateAllTrees(
        const NMatrixnet::TMnSseStaticMeta& info,
        const typename TFetcher<WordLen>::TLeaf* data,
        const ui32 *indices,
        TFetcher<WordLen>* fetcher,
        const ui32* val
) {
    const ui32 *end = info.DataIndicesSize + indices;
    int proceed = 0;
    if constexpr (FetcherType == FT_ShuffleFetch) {
        data = (int*)((char*)data + (0x40 - (uintptr_t)data % 0x40));
    }

    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 0>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 1>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 2>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 3>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 4>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 5>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 6>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 7>(info, data, indices, end, fetcher, val, proceed);
    CalculateAllTreesOnDepth<Num, WordLen, Ordered, FetcherType, 8>(info, data, indices, end, fetcher, val, proceed);
}

template<size_t Num, size_t WordLen, bool Ordered, TFetcherType FetcherType>
static void AvxApply(const TPreparedSubBatch& preparedSubBatch, TFetcher<WordLen>* fetcher, const NMatrixnet::TMnSseStatic &info, double* res) {
    const TVector<NMatrixnet::TMultiData::TLeafData>& multiData = Get<NMatrixnet::TMultiData>(info.Leaves.Data).MultiData;
    Y_ENSURE(multiData.size() == 1);
    const ui32 *dataIndices = (const ui32*) info.Meta.DataIndicesPtr;

    CalculateAllTrees<Num, WordLen, Ordered, FetcherType>(info.Meta, multiData[0].Data, dataIndices,
                                                          fetcher, preparedSubBatch.GetVal());
    fetcher->template DoFlush<Num>();

    ui64 sub = info.Meta.GetSizeToCount(0) +
               info.Meta.GetSizeToCount(1) +
               info.Meta.GetSizeToCount(2) +
               info.Meta.GetSizeToCount(3) +
               info.Meta.GetSizeToCount(4) +
               info.Meta.GetSizeToCount(5) +
               info.Meta.GetSizeToCount(6) +
               info.Meta.GetSizeToCount(7) +
               info.Meta.GetSizeToCount(8);
    sub <<= 31;

    const double dataScale = multiData[0].Norm.DataScale;
    const double dataBias = multiData[0].Norm.DataBias;
    const i64* src = (i64*)fetcher->Vals;
    if constexpr (Ordered || WordLen == 128 || FetcherType == FT_GaitherFetch) {
        for (size_t i = 0; i < preparedSubBatch.NumFactors; ++i) {
            res[i] = (src[i] - sub * 1.0) * dataScale + dataBias;
        }
    } else {
        const size_t SubsubbatchSize = WordLen / 8;
        double tmp[SubsubbatchSize] = {};
        for (size_t i = 0; i < Num; ++i) {
            const size_t CurSubsubbatchSize = (i + 1 != Num ? SubsubbatchSize : preparedSubBatch.NumFactors - i * SubsubbatchSize);
            for (size_t j = 0; j < SubsubbatchSize; ++j) {
                tmp[j % 4 + j / 4 * (SubsubbatchSize / 4) % SubsubbatchSize + j / 4 * (SubsubbatchSize / 4) / SubsubbatchSize * 4]
                        = (src[i * SubsubbatchSize + j] - sub * 1.0) * dataScale + dataBias;
            }
            std::memcpy(res + i * SubsubbatchSize, tmp, sizeof(double) * CurSubsubbatchSize);
        }
    }
}

template<size_t WordLen, bool Ordered, TFetcherType FetcherType>
void AvxLongApply(const TAvxBinarization<WordLen, Ordered>& preparedBatch, const NMatrixnet::TMnSseStatic &info, double* res) {
    Y_ENSURE(!info.Meta.Has16Bits);
    for (size_t i = 0; i < preparedBatch.Batches.size(); ++i) {
        size_t numFetch = 0;
        if (Ordered || WordLen == 128 || FetcherType == FT_GaitherFetch) {
            numFetch = preparedBatch.Batches[i].NumFactors;
        } else {
            numFetch = ((preparedBatch.Batches[i].NumFactors - 1) / (WordLen / 8) + 1) * (WordLen / 8);
        }
        TFetcher<WordLen> fetcher(numFetch);
        switch ((preparedBatch.Batches[i].NumFactors - 1) / (WordLen / 8)) {
            case 0:
                AvxApply<1, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
            case 1:
                AvxApply<2, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
            case 2:
                AvxApply<3, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
            case 3:
                AvxApply<4, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
            case 4:
                AvxApply<5, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
            case 5:
                AvxApply<6, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
            case 6:
                AvxApply<7, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
            case 7:
                AvxApply<8, WordLen, Ordered, FetcherType>(preparedBatch.Batches[i], &fetcher, info, res + i * WordLen);
                break;
        }
    }
}

template<size_t WordLen, bool Ordered, TFetcherType FetcherType>
TVector<double> AvxApply(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<WordLen, Ordered>& preparedBatch) {
    TVector<double> res(preparedBatch.GetNumDocs());
    AvxLongApply<WordLen, Ordered, FetcherType>(preparedBatch, info, res.data());
    return res;
}

TVector<double> Avx128Apply(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, false> &preparedBatch) {
    return AvxApply<128, false, FT_FetchFourTrees>(info, preparedBatch);
}

TVector<double> Avx256Apply(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, false> &preparedBatch) {
    return AvxApply<256, false, FT_FetchFourTrees>(info, preparedBatch);
}

TVector<double> Avx512Apply(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch) {
    return AvxApply<512, false, FT_FetchFourTrees>(info, preparedBatch);
}

TVector<double> Avx128ApplyOrdered(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, true> &preparedBatch) {
    return AvxApply<128, true, FT_FetchFourTrees>(info, preparedBatch);
}

TVector<double> Avx256ApplyOrdered(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, true> &preparedBatch) {
    return AvxApply<256, true, FT_FetchFourTrees>(info, preparedBatch);
}

TVector<double> Avx512ApplyOrdered(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, true> &preparedBatch) {
    return AvxApply<512, true, FT_FetchFourTrees>(info, preparedBatch);
}

TVector<double> Avx128ApplyOneTreeFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, false> &preparedBatch) {
    return AvxApply<128, false, FT_FetchOneTree>(info, preparedBatch);
}

TVector<double> Avx256ApplyOneTreeFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, false> &preparedBatch) {
    return AvxApply<256, false, FT_FetchOneTree>(info, preparedBatch);
}

TVector<double> Avx512ApplyOneTreeFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch) {
    return AvxApply<512, false, FT_FetchOneTree>(info, preparedBatch);
}

TVector<double> Avx128ApplyInstantFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, false> &preparedBatch) {
    return AvxApply<128, false, FT_GaitherFetch>(info, preparedBatch);
}

TVector<double> Avx256ApplyInstantFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, false> &preparedBatch) {
    return AvxApply<256, false, FT_GaitherFetch>(info, preparedBatch);
}

TVector<double> Avx512ApplyInstantFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch) {
    return AvxApply<512, false, FT_GaitherFetch>(info, preparedBatch);
}

TVector<double> Avx512ApplyShuffleFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch) {
    return AvxApply<512, false, FT_ShuffleFetch>(info, preparedBatch);
}
