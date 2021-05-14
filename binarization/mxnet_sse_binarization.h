#pragma once

#include <binarization_formats/mxnet_sse_format.h>
#include <immintrin.h>

namespace {
using namespace NMatrixnet;

static __m128i NFORCED_INLINE GetCmp16(const __m128 &c0, const __m128 &c1, const __m128 &c2, const __m128 &c3, const __m128 test) {
    const __m128i r0 = _mm_castps_si128(_mm_cmpgt_ps(c0, test));
    const __m128i r1 = _mm_castps_si128(_mm_cmpgt_ps(c1, test));
    const __m128i r2 = _mm_castps_si128(_mm_cmpgt_ps(c2, test));
    const __m128i r3 = _mm_castps_si128(_mm_cmpgt_ps(c3, test));
    const __m128i packed = _mm_packs_epi16(_mm_packs_epi32(r0, r1), _mm_packs_epi32(r2, r3));
    return _mm_and_si128(_mm_set1_epi8(0x01), packed);
}

static __m128i NFORCED_INLINE GetCmp16(const float *factors, const __m128 test) {
    const __m128 *ptr = (__m128 *)factors;
    return GetCmp16(ptr[0], ptr[1], ptr[2], ptr[3], test);
}

static NFORCED_INLINE void Group4(float *dst, const float* const factors[128], size_t index0, size_t index1, size_t index2, size_t index3, size_t num, const __m128& substitutionVal) {
    for (size_t i = 0; i < num; ++i) {
        const float *s = factors[i];
        const __m128 tmpvals = _mm_set_ps(s[index3], s[index2], s[index1], s[index0]);
        const __m128 masks = _mm_cmpunord_ps(tmpvals, tmpvals);
        const __m128 result = _mm_or_ps(_mm_andnot_ps(masks, tmpvals), _mm_and_ps(masks, substitutionVal));
        _mm_store_ss(&dst[i + 0 * 128], result);
        _mm_store_ss(&dst[i + 1 * 128], _mm_shuffle_ps(result,result, _MM_SHUFFLE(0,0,0,1)));
        _mm_store_ss(&dst[i + 2 * 128], _mm_shuffle_ps(result,result, _MM_SHUFFLE(0,0,0,2)));
        _mm_store_ss(&dst[i + 3 * 128], _mm_shuffle_ps(result,result, _MM_SHUFFLE(0,0,0,3)));
    }
}

static NFORCED_INLINE void Group4(float *dst, const float* const factors[128], size_t index0, size_t index1, size_t index2, size_t index3, size_t num) {
    for (size_t i = 0; i < num; ++i) {
        const float *factor = factors[i];
        dst[i + 0 * 128] = factor[index0];
        dst[i + 1 * 128] = factor[index1];
        dst[i + 2 * 128] = factor[index2];
        dst[i + 3 * 128] = factor[index3];
    }
}

template<size_t Num>
static NFORCED_INLINE void DoLane(size_t length, const float *factors, ui32 *&dst, const float *&values) {
    for (size_t i = 0; i < length; ++i) {
        __m128 value = _mm_set1_ps(values[0]);
        __m128i agg =                               GetCmp16(factors + 0 * 16, value);
        if (Num > 1)
            agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(factors + 1 * 16, value), 1));
        if (Num > 2)
            agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(factors + 2 * 16, value), 2));
        if (Num > 3)
            agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(factors + 3 * 16, value), 3));
        if (Num > 4)
            agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(factors + 4 * 16, value), 4));
        if (Num > 5)
            agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(factors + 5 * 16, value), 5));
        if (Num > 6)
            agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(factors + 6 * 16, value), 6));
        if (Num > 7)
            agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(factors + 7 * 16, value), 7));
        _mm_store_si128((__m128i *)dst, agg);
        dst += 4;
        ++values;
    }
}

template <size_t X>
void CalcFactors(const TFeature *features, const i8* missedValueDirections, const float *compares, size_t featureLength, const float* const factors[128], size_t num, ui32 *out, const ui32 featsIndexOff = 0) {
    alignas(0x10) float factorsGrouped[4][128];
    memset(factorsGrouped, 0, sizeof(factorsGrouped));
    const size_t last = featureLength ? featureLength - 1 : 0;
    const size_t remainder = featureLength % 4;
    const size_t featureLength4 = featureLength - remainder;
    size_t i = 0;
    const auto infinity = std::numeric_limits<float>::infinity(); // use alias just to keep code smaller
    Y_ENSURE(!missedValueDirections);
    for (; i < featureLength4; i += 4) {
        Group4(
            &factorsGrouped[0][0],
            factors,
            features[i + 0].Index - featsIndexOff,
            features[i + 1].Index - featsIndexOff,
            features[i + 2].Index - featsIndexOff,
            features[i + 3].Index - featsIndexOff,
            num
        );
        DoLane<X>(features[i + 0].Length, &factorsGrouped[0][0], out, compares);
        DoLane<X>(features[i + 1].Length, &factorsGrouped[1][0], out, compares);
        DoLane<X>(features[i + 2].Length, &factorsGrouped[2][0], out, compares);
        DoLane<X>(features[i + 3].Length, &factorsGrouped[3][0], out, compares);
    }

    if (remainder > 0) {
        Group4(
            &factorsGrouped[0][0],
            factors,
            features[Min(i + 0, last)].Index - featsIndexOff,
            features[Min(i + 1, last)].Index - featsIndexOff,
            features[Min(i + 2, last)].Index - featsIndexOff,
            features[Min(i + 3, last)].Index - featsIndexOff,
            num);
        for (size_t j = 0; j < remainder; ++j) {
            DoLane<X>(features[i + j].Length, &factorsGrouped[j][0], out, compares);
        }
    }
}


template <size_t Num>
static void CalculateFeatures(const TMnSseStaticMeta& info, const float* const* factors, size_t numDocs, size_t stride, size_t numSlices, ui32* val) {
    CalcFactors<Num>(info.Features, info.MissedValueDirections, info.Values, info.FeaturesSize, factors, numDocs, val);
}

TMxNetSseBinarization MxNetLongBinarization(const NMatrixnet::TMnSseStaticMeta &mt, const float* const* factors, size_t num, size_t stride, size_t numSlices) {
    TMxNetSseBinarization preparedBatch(mt, num);
    for (size_t i = 0; i < preparedBatch.Batches.size(); ++i) {
        switch ((preparedBatch.Batches[i].NumFactors - 1) / 16) {
            case 0:
                CalculateFeatures<1>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
            case 1:
                CalculateFeatures<2>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
            case 2:
                CalculateFeatures<3>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
            case 3:
                CalculateFeatures<4>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
            case 4:
                CalculateFeatures<5>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
            case 5:
                CalculateFeatures<6>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
            case 6:
                CalculateFeatures<7>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
            case 7:
                CalculateFeatures<8>(mt, factors + i * 128, preparedBatch.Batches[i].NumFactors, stride, numSlices, preparedBatch.Batches[i].GetVal());
                break;
        }
    }
    return preparedBatch;
}
}

TMxNetSseBinarization MxNetSseBinarization(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>> &features) {
    TVector<const float*> fPtrs(features.size());
    for (int i = 0; i < features.size(); ++i) {
        fPtrs[i] = (features[i]).data();
    }
    return MxNetLongBinarization(mn, fPtrs.data(), features.size(), features.size(), 1);
}
