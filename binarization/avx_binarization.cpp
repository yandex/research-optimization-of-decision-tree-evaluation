#include "avx_binarization.h"
#include <util/util.h>
#include <util/model.h>
#include <immintrin.h>


static ui64 NFORCED_INLINE GetCmpOrdered64_512(const __m512 &c0, const __m512 &c1, const __m512 &c2, const __m512 &c3, const __m512 test) {
    const __mmask16 r0 = _mm512_cmpnle_ps_mask(c0, test);
    const __mmask16 r1 = _mm512_cmpnle_ps_mask(c1, test);
    const __mmask16 r2 = _mm512_cmpnle_ps_mask(c2, test);
    const __mmask16 r3 = _mm512_cmpnle_ps_mask(c3, test);
    return (ui64)_cvtmask16_u32(r0) | ((ui64)_cvtmask16_u32(r1) << 16) | ((ui64)_cvtmask16_u32(r2) << 32) | ((ui64)_cvtmask16_u32(r3) << 48);
}

static ui64 NFORCED_INLINE GetCmpOrdered64_512(const float *factors, const __m512 test) {
    const __m512 *ptr = (__m512 *)factors;
    return GetCmpOrdered64_512(ptr[0], ptr[1], ptr[2], ptr[3], test);
}

static ui32 NFORCED_INLINE GetCmpOrdered32_256(const __m256 &c0, const __m256 &c1, const __m256 &c2, const __m256 &c3, const __m256 test) {
    const __mmask8 r0 = _mm256_cmp_ps_mask(c0, test, _CMP_GT_OQ);
    const __mmask8 r1 = _mm256_cmp_ps_mask(c1, test, _CMP_GT_OQ);
    const __mmask8 r2 = _mm256_cmp_ps_mask(c2, test, _CMP_GT_OQ);
    const __mmask8 r3 = _mm256_cmp_ps_mask(c3, test, _CMP_GT_OQ);
    return _cvtmask8_u32(r0) | (_cvtmask8_u32(r1) << 8) | (_cvtmask8_u32(r2) << 16) | (_cvtmask8_u32(r3) << 24);
}

static ui32 NFORCED_INLINE GetCmpOrdered32_256(const float *factors, const __m256 test) {
    const __m256 *ptr = (__m256 *)factors;
    return GetCmpOrdered32_256(ptr[0], ptr[1], ptr[2], ptr[3], test);
}

static ui16 NFORCED_INLINE GetCmpOrdered16_128(const __m128 &c0, const __m128 &c1, const __m128 &c2, const __m128 &c3, const __m128 test) {
    const __mmask8 r0 = _mm_cmp_ps_mask(c0, test, _CMP_GT_OQ);
    const __mmask8 r1 = _mm_cmp_ps_mask(c1, test, _CMP_GT_OQ);
    const __mmask8 r2 = _mm_cmp_ps_mask(c2, test, _CMP_GT_OQ);
    const __mmask8 r3 = _mm_cmp_ps_mask(c3, test, _CMP_GT_OQ);
    return _cvtmask8_u32(r0) | (_cvtmask8_u32(r1) << 4) | (_cvtmask8_u32(r2) << 8) | (_cvtmask8_u32(r3) << 12);
}

static ui16 NFORCED_INLINE GetCmpOrdered16_128(const float *factors, const __m128 test) {
    const __m128 *ptr = (__m128 *)factors;
    return GetCmpOrdered16_128(ptr[0], ptr[1], ptr[2], ptr[3], test);
}

template<size_t Num, size_t WordLen>
static void DoLaneOrdered(size_t length, const float *factors, ui32 *&dst, const float *&values) {
    if constexpr (WordLen == 512) {
        for (size_t i = 0; i < length; ++i) {
            __m512 value = _mm512_set1_ps(values[0]);
            NFORCE_UNROLL
            for (size_t j = 0; j < Num; ++j) {
                reinterpret_cast<ui64*>(dst)[j] = GetCmpOrdered64_512(factors + j * 64, value);
            }
            dst += WordLen / sizeof(ui32) / 8;
            ++values;
        }
    } else if constexpr (WordLen == 256) {
        for (size_t i = 0; i < length; ++i) {
            __m256 value = _mm256_set1_ps(values[0]);
            NFORCE_UNROLL
            for (size_t j = 0; j < Num; ++j) {
                dst[j] = GetCmpOrdered32_256(factors + j * 32, value);
            }
            dst += WordLen / sizeof(ui32) / 8;
            ++values;
        }
    } else if constexpr (WordLen == 128) {
        for (size_t i = 0; i < length; ++i) {
            __m128 value = _mm_set1_ps(values[0]);
            NFORCE_UNROLL
            for (size_t j = 0; j < Num; ++j) {
                reinterpret_cast<ui16*>(dst)[j] = GetCmpOrdered16_128(factors + j * 16, value);
            }
            dst += WordLen / sizeof(ui32) / 8;
            ++values;
        }
    } else {
        Y_FAIL("Unsupported wordlen");
    }
}

static __m512i NFORCED_INLINE GetCmp64_512(const __m512 &c0, const __m512 &c1, const __m512 &c2, const __m512 &c3, const __m512 test) {
    const __m512i r0 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(c0, test, _CMP_GT_OQ), 1);
    const __m512i r1 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(c1, test, _CMP_GT_OQ), 1);
    const __m512i r2 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(c2, test, _CMP_GT_OQ), 1);
    const __m512i r3 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(c3, test, _CMP_GT_OQ), 1);
    return _mm512_packs_epi16(_mm512_packs_epi32(r0, r1), _mm512_packs_epi32(r2, r3));
}

static __m512i NFORCED_INLINE GetCmp64_512(const float *factors, const __m512 test) {
    const __m512 *ptr = (__m512 *)factors;
    return GetCmp64_512(ptr[0], ptr[1], ptr[2], ptr[3], test);
}

static __m256i NFORCED_INLINE GetCmp32_256(const __m256 &c0, const __m256 &c1, const __m256 &c2, const __m256 &c3, const __m256 test) {
    const __m256i r0 = _mm256_castps_si256(_mm256_cmp_ps(c0, test, _CMP_GT_OQ));
    const __m256i r1 = _mm256_castps_si256(_mm256_cmp_ps(c1, test, _CMP_GT_OQ));
    const __m256i r2 = _mm256_castps_si256(_mm256_cmp_ps(c2, test, _CMP_GT_OQ));
    const __m256i r3 = _mm256_castps_si256(_mm256_cmp_ps(c3, test, _CMP_GT_OQ));
    const __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(r0, r1), _mm256_packs_epi32(r2, r3));
    return _mm256_and_si256(_mm256_set1_epi8(0x01), packed);
}

static __m256i NFORCED_INLINE GetCmp32_256(const float *factors, const __m256 test) {
    const __m256 *ptr = (__m256 *)factors;
    return GetCmp32_256(ptr[0], ptr[1], ptr[2], ptr[3], test);
}

static __m128i NFORCED_INLINE GetCmp16_128(const __m128 &c0, const __m128 &c1, const __m128 &c2, const __m128 &c3, const __m128 test) {
    const __m128i r0 = _mm_castps_si128(_mm_cmpgt_ps(c0, test));
    const __m128i r1 = _mm_castps_si128(_mm_cmpgt_ps(c1, test));
    const __m128i r2 = _mm_castps_si128(_mm_cmpgt_ps(c2, test));
    const __m128i r3 = _mm_castps_si128(_mm_cmpgt_ps(c3, test));
    const __m128i packed = _mm_packs_epi16(_mm_packs_epi32(r0, r1), _mm_packs_epi32(r2, r3));
    return _mm_and_si128(_mm_set1_epi8(0x01), packed);
}

static __m128i NFORCED_INLINE GetCmp16_128(const float *factors, const __m128 test) {
    const __m128 *ptr = (__m128 *)factors;
    return GetCmp16_128(ptr[0], ptr[1], ptr[2], ptr[3], test);
}

template<size_t Num, size_t WordLen>
static void DoLane(size_t length, const float *factors, ui32 *&dst, const float *&values) {
    if constexpr (WordLen == 512) {
        for (size_t i = 0; i < length; ++i) {
            __m512 value = _mm512_set1_ps(values[0]);
            __m512i agg =                                    GetCmp64_512(factors + 0 * 64, value);
            NFORCE_UNROLL
            for (size_t i = 1; i < Num; ++i) {
                agg = _mm512_or_si512(agg, _mm512_slli_epi16(GetCmp64_512(factors + i * 64, value), i));
            }
            _mm512_storeu_si512(dst, agg);
            dst += WordLen / sizeof(ui32) / 8;
            ++values;
        }
    } else if constexpr (WordLen == 256) {
        for (size_t i = 0; i < length; ++i) {
            __m256 value = _mm256_set1_ps(values[0]);
            __m256i agg =                                    GetCmp32_256(factors + 0 * 32, value);
            NFORCE_UNROLL
            for (size_t i = 1; i < Num; ++i) {
                agg = _mm256_or_si256(agg, _mm256_slli_epi16(GetCmp32_256(factors + i * 32, value), i));
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), agg);
            dst += WordLen / sizeof(ui32) / 8;
            ++values;
        }
    } else if constexpr (WordLen == 128) {
        for (size_t i = 0; i < length; ++i) {
            __m128 value = _mm_set1_ps(values[0]);
            __m128i agg =                              GetCmp16_128(factors + 0 * 16, value);
            NFORCE_UNROLL
            for (size_t i = 1; i < Num; ++i) {
                agg = _mm_or_si128(agg, _mm_slli_epi16(GetCmp16_128(factors + i * 16, value), i));
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), agg);
            dst += WordLen / sizeof(ui32) / 8;
            ++values;
        }
    } else {
        Y_FAIL("Unsupported wordlen");
    }
}

template<size_t WordLen>
static void Group16(float *dst, const float* const factors[WordLen], size_t index0, size_t index1, size_t index2, size_t index3, size_t index4, size_t index5, size_t index6, size_t index7, size_t index8, size_t index9, size_t index10, size_t index11, size_t index12, size_t index13, size_t index14, size_t index15, size_t num) {
    for (size_t i = 0; i < num; ++i) {
        const float *factor = factors[i];
        dst[i + 0  * WordLen] = factor[index0];
        dst[i + 1  * WordLen] = factor[index1];
        dst[i + 2  * WordLen] = factor[index2];
        dst[i + 3  * WordLen] = factor[index3];
        dst[i + 4  * WordLen] = factor[index4];
        dst[i + 5  * WordLen] = factor[index5];
        dst[i + 6  * WordLen] = factor[index6];
        dst[i + 7  * WordLen] = factor[index7];
        dst[i + 8  * WordLen] = factor[index8];
        dst[i + 9  * WordLen] = factor[index9];
        dst[i + 10 * WordLen] = factor[index10];
        dst[i + 11 * WordLen] = factor[index11];
        dst[i + 12 * WordLen] = factor[index12];
        dst[i + 13 * WordLen] = factor[index13];
        dst[i + 14 * WordLen] = factor[index14];
        dst[i + 15 * WordLen] = factor[index15];
    }
}

template <size_t X, size_t WordLen, bool Ordered>
void CalcFactors(const NMatrixnet::TFeature *features, const float *compares, size_t featureLength, const float* const factors[WordLen], size_t num, ui32 *out, const ui32 featsIndexOff = 0) {
    alignas(0x40) float factorsGrouped[16][WordLen];
    memset(factorsGrouped, 0, sizeof(factorsGrouped));
    const size_t last = featureLength ? featureLength - 1 : 0;
    const size_t remainder = featureLength % 16;
    size_t i = 0;
    for (; i < featureLength - remainder; i += 16) {
        Group16<WordLen>(
            &factorsGrouped[0][0],
            factors,
            features[i +  0].Index - featsIndexOff,
            features[i +  1].Index - featsIndexOff,
            features[i +  2].Index - featsIndexOff,
            features[i +  3].Index - featsIndexOff,
            features[i +  4].Index - featsIndexOff,
            features[i +  5].Index - featsIndexOff,
            features[i +  6].Index - featsIndexOff,
            features[i +  7].Index - featsIndexOff,
            features[i +  8].Index - featsIndexOff,
            features[i +  9].Index - featsIndexOff,
            features[i + 10].Index - featsIndexOff,
            features[i + 11].Index - featsIndexOff,
            features[i + 12].Index - featsIndexOff,
            features[i + 13].Index - featsIndexOff,
            features[i + 14].Index - featsIndexOff,
            features[i + 15].Index - featsIndexOff,
            num
        );
        NFORCE_UNROLL
        for (size_t j = 0; j < 16; ++j) {
            if constexpr (Ordered) {
                DoLaneOrdered<X, WordLen>(features[i + j].Length, &factorsGrouped[j][0], out, compares);
            } else {
                DoLane       <X, WordLen>(features[i + j].Length, &factorsGrouped[j][0], out, compares);
            }
        }
    }

    if (remainder > 0) {
        Group16<WordLen>(
            &factorsGrouped[0][0],
            factors,
            features[Min(i +  0, last)].Index - featsIndexOff,
            features[Min(i +  1, last)].Index - featsIndexOff,
            features[Min(i +  2, last)].Index - featsIndexOff,
            features[Min(i +  3, last)].Index - featsIndexOff,
            features[Min(i +  4, last)].Index - featsIndexOff,
            features[Min(i +  5, last)].Index - featsIndexOff,
            features[Min(i +  6, last)].Index - featsIndexOff,
            features[Min(i +  7, last)].Index - featsIndexOff,
            features[Min(i +  8, last)].Index - featsIndexOff,
            features[Min(i +  9, last)].Index - featsIndexOff,
            features[Min(i + 10, last)].Index - featsIndexOff,
            features[Min(i + 11, last)].Index - featsIndexOff,
            features[Min(i + 12, last)].Index - featsIndexOff,
            features[Min(i + 13, last)].Index - featsIndexOff,
            features[Min(i + 14, last)].Index - featsIndexOff,
            features[Min(i + 15, last)].Index - featsIndexOff,
            num);
        for (size_t j = 0; j < remainder; ++j) {
            if constexpr (Ordered) {
                DoLaneOrdered<X, WordLen>(features[i + j].Length, &factorsGrouped[j][0], out, compares);
            } else {
                DoLane       <X, WordLen>(features[i + j].Length, &factorsGrouped[j][0], out, compares);
            }
        }
    }
}

template <size_t Num, size_t WordLen, bool Oredered = false>
static void CalculateFeatures(const NMatrixnet::TMnSseStaticMeta& info, const float* const* factors, size_t numDocs, ui32* val) {
    Y_ENSURE(!info.MissedValueDirections);
    CalcFactors<Num, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, numDocs, val);
}

template<size_t WordLen, bool Oredered>
TAvxBinarization<WordLen, Oredered> AvxLongBinarization(const NMatrixnet::TMnSseStaticMeta &mt, const float* const* factors, size_t num) {
    TAvxBinarization<WordLen, Oredered> preparedBatch(mt, num);
    for (size_t i = 0; i < preparedBatch.Batches.size(); ++i) {
        switch ((preparedBatch.Batches[i].NumFactors - 1) / (WordLen / 8)) {
            case 0:
                CalculateFeatures<1, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
            case 1:
                CalculateFeatures<2, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
            case 2:
                CalculateFeatures<3, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
            case 3:
                CalculateFeatures<4, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
            case 4:
                CalculateFeatures<5, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
            case 5:
                CalculateFeatures<6, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
            case 6:
                CalculateFeatures<7, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
            case 7:
                CalculateFeatures<8, WordLen, Oredered>(mt, factors + i * WordLen, preparedBatch.Batches[i].NumFactors, preparedBatch.Batches[i].GetVal());
                break;
        }
    }
    return preparedBatch;
}


template<size_t WordLen, bool Oredered>
TAvxBinarization<WordLen, Oredered> AvxBinarization(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    TVector<const float*> fPtrs(features.size());
    for (int i = 0; i < features.size(); ++i) {
        fPtrs[i] = (features[i]).data();
    }
    return AvxLongBinarization<WordLen, Oredered>(mn, fPtrs.data(), features.size());
}

TAvxBinarization<512, true> Avx512BinarizationOrdered(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    return AvxBinarization<512, true>(mn, features);
}

TAvxBinarization<256, true> Avx256BinarizationOrdered(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    return AvxBinarization<256, true>(mn, features);
}

TAvxBinarization<128, true> Avx128BinarizationOrdered(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    return AvxBinarization<128, true>(mn, features);
}

TAvxBinarization<512, false> Avx512Binarization(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    return AvxBinarization<512, false>(mn, features);
}

TAvxBinarization<256, false> Avx256Binarization(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    return AvxBinarization<256, false>(mn, features);
}

TAvxBinarization<128, false> Avx128Binarization(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    return AvxBinarization<128, false>(mn, features);
}

template <size_t X, size_t WordLen, bool Ordered>
void CalcFactorsTransposed(const NMatrixnet::TFeature *features, const float *compares, size_t featureLength, const TVector<float*>& factors, size_t offset, ui32 *out) {
    for (size_t i = 0; i < featureLength; ++i) {
        if constexpr (Ordered) {
            DoLaneOrdered<X, WordLen>(features[i].Length, &factors[i][offset], out, compares);
        } else {
            DoLane       <X, WordLen>(features[i].Length, &factors[i][offset], out, compares);
        }
    }
}

template<size_t WordLen, bool Oredered>
TAvxBinarization<WordLen, Oredered> AvxLongBinarizationTransposed(const NMatrixnet::TMnSseStaticMeta &info, const TVector<float*>& factors, size_t numDocs) {
    TAvxBinarization<WordLen, Oredered> preparedBatch(info, numDocs);
    for (size_t i = 0; i < preparedBatch.Batches.size(); ++i) {
        switch ((preparedBatch.Batches[i].NumFactors - 1) / (WordLen / 8)) {
            case 0:
                CalcFactorsTransposed<1, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
            case 1:
                CalcFactorsTransposed<2, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
            case 2:
                CalcFactorsTransposed<3, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
            case 3:
                CalcFactorsTransposed<4, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
            case 4:
                CalcFactorsTransposed<5, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
            case 5:
                CalcFactorsTransposed<6, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
            case 6:
                CalcFactorsTransposed<7, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
            case 7:
                CalcFactorsTransposed<8, WordLen, Oredered>(info.Features, info.Values, info.FeaturesSize, factors, i * WordLen, preparedBatch.Batches[i].GetVal());
                break;
        }
    }
    return preparedBatch;
}

TAvxBinarization<512, true> Avx512BinarizationOrderedTransposed(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<float*> &features, size_t numDocs) {
    return AvxLongBinarizationTransposed<512, true>(mn, features, numDocs);
}

TAvxBinarization<256, true> Avx256BinarizationOrderedTransposed(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<float*> &features, size_t numDocs) {
    return AvxLongBinarizationTransposed<256, true>(mn, features, numDocs);
}

TAvxBinarization<128, true> Avx128BinarizationOrderedTransposed(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<float*> &features, size_t numDocs) {
    return AvxLongBinarizationTransposed<128, true>(mn, features, numDocs);
}

TAvxBinarization<512, false> Avx512BinarizationTransposed(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<float*> &features, size_t numDocs) {
    return AvxLongBinarizationTransposed<512, false>(mn, features, numDocs);
}

TAvxBinarization<256, false> Avx256BinarizationTransposed(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<float*> &features, size_t numDocs) {
    return AvxLongBinarizationTransposed<256, false>(mn, features, numDocs);
}

TAvxBinarization<128, false> Avx128BinarizationTransposed(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<float*> &features, size_t numDocs) {
    return AvxLongBinarizationTransposed<128, false>(mn, features, numDocs);
}
