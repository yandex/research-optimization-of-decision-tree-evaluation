#pragma once

#include <binarization_formats/mxnet_sse_format.h>

namespace {
template<int X>
struct TGetSign {
    static constexpr int Value = (X > 0) ? 1 : ((X < 0) ? -1 : 0);
};

template<int X, int Shift, int ShiftSign>
struct TSelBitImpl;

template<int X, int Shift>
struct TSelBitImpl<X, Shift, 1> {
    static NFORCED_INLINE __m128i SelBit(__m128i value) {
        return _mm_and_si128(_mm_slli_epi16(value, +Shift), _mm_slli_epi16(_mm_set1_epi8(0x01), X));
    }
};

template<int X, int Shift>
struct TSelBitImpl<X, Shift, -1> {
    static NFORCED_INLINE __m128i SelBit(__m128i value) {
        return _mm_and_si128(_mm_srli_epi16(value, -Shift), _mm_slli_epi16(_mm_set1_epi8(0x01), X));
    }
};

template<int X>
struct TSelBitImpl<X, 0, 0> {
    static NFORCED_INLINE __m128i SelBit(__m128i value) {
        return _mm_and_si128(value, _mm_slli_epi16(_mm_set1_epi8(0x01), X));
    }
};


template<int X, int Shift>
static NFORCED_INLINE __m128i SelBit(__m128i value) {
    return TSelBitImpl<X, Shift, TGetSign<Shift>::Value>::SelBit(value);
}

template<int X>
static void NFORCED_INLINE AssignAdd(__m128i &v, __m128i add) {
    if (X != 0)
        v = _mm_add_epi32(v, add);
    else
        v = add;
}

template<size_t Depth, int X>
static void NFORCED_INLINE DoBitSet(
    __m128i &v0,
    __m128i &v1,
    __m128i &v2,
    __m128i &v3,
    __m128i &v4,
    __m128i &v5,
    __m128i &v6,
    __m128i &v7,
    __m128i b,
    const void *fetch)
{
    Y_PREFETCH_READ(fetch, 3);
    AssignAdd<X>(v0, SelBit<X, X - 0>(b));
    AssignAdd<X>(v1, SelBit<X, X - 1>(b));
    AssignAdd<X>(v2, SelBit<X, X - 2>(b));
    AssignAdd<X>(v3, SelBit<X, X - 3>(b));
    AssignAdd<X>(v4, SelBit<X, X - 4>(b));
    AssignAdd<X>(v5, SelBit<X, X - 5>(b));
    AssignAdd<X>(v6, SelBit<X, X - 6>(b));
    AssignAdd<X>(v7, SelBit<X, X - 7>(b));
}

template<size_t Num, size_t Depth, typename T>
static void DoMXLane(const ui32 *val, const T *indices, const T *end, ui8 *res) {
    const T *fetch = Min(end - Depth, indices + 10);
    __m128i v0 = _mm_setzero_si128();
    __m128i v1 = _mm_setzero_si128();
    __m128i v2 = _mm_setzero_si128();
    __m128i v3 = _mm_setzero_si128();
    __m128i v4 = _mm_setzero_si128();
    __m128i v5 = _mm_setzero_si128();
    __m128i v6 = _mm_setzero_si128();
    __m128i v7 = _mm_setzero_si128();

    if (0 < Depth) {
        DoBitSet<Depth, 0>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[0])), val + fetch[0]);
    }
    if (1 < Depth) {
        DoBitSet<Depth, 1>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[1])), val + fetch[1]);
    }
    if (2 < Depth) {
        DoBitSet<Depth, 2>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[2])), val + fetch[2]);
    }
    if (3 < Depth) {
        DoBitSet<Depth, 3>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[3])), val + fetch[3]);
    }
    if (4 < Depth) {
        DoBitSet<Depth, 4>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[4])), val + fetch[4]);
    }
    if (5 < Depth) {
        DoBitSet<Depth, 5>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[5])), val + fetch[5]);
    }
    if (6 < Depth) {
        DoBitSet<Depth, 6>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[6])), val + fetch[6]);
    }
    if (7 < Depth) {
        DoBitSet<Depth, 7>(v0, v1, v2, v3, v4, v5, v6, v7, _mm_load_si128((__m128i *)(val + indices[7])), val + fetch[7]);
    }

    Y_PREFETCH_READ(indices, 3);
    if (Num > 0)
        _mm_store_si128((__m128i *)(res + 16 * 0), v0);
    if (Num > 1)
        _mm_store_si128((__m128i *)(res + 16 * 1), v1);
    if (Num > 2)
        _mm_store_si128((__m128i *)(res + 16 * 2), v2);
    if (Num > 3)
        _mm_store_si128((__m128i *)(res + 16 * 3), v3);
    if (Num > 4)
        _mm_store_si128((__m128i *)(res + 16 * 4), v4);
    if (Num > 5)
        _mm_store_si128((__m128i *)(res + 16 * 5), v5);
    if (Num > 6)
        _mm_store_si128((__m128i *)(res + 16 * 6), v6);
    if (Num > 7)
        _mm_store_si128((__m128i *)(res + 16 * 7), v7);
}

static void  NFORCED_INLINE DoFetch2(ui16 value, const int *d0, __m128i &val) {
    const __m128i lo = _mm_cvtsi32_si128(d0[value & 0xff]);
    const __m128i hi = _mm_cvtsi32_si128(d0[value >> 8]);
    val = _mm_add_epi64(val, _mm_unpacklo_epi64(lo, hi));
}

template <typename TLeaf, typename TDocValues, size_t X>
struct TFetcherWorker;

template<size_t... I>
static NFORCED_INLINE void DoFetchMany(__m128i* datas, const int *d0, const int *d1, const int *d2, const int *d3, const ui16 *bins, const std::index_sequence<I...>&) {
    __m128i val;
    auto dummy = {
        (
        val = datas[I],
        DoFetch2(bins[I + 0 * 64], d0, val),
        DoFetch2(bins[I + 1 * 64], d1, val),
        DoFetch2(bins[I + 2 * 64], d2, val),
        DoFetch2(bins[I + 3 * 64], d3, val),
        datas[I] = val)...};
    Y_UNUSED(dummy);
}

template <size_t X>
struct TFetcherWorker<int, ui64[128], X> {
    static void DoFetch128(const int *d0, const int *d1, const int *d2, const int *d3, const ui16 *bins, ui64 (*datasOrig)[128], size_t num) {
        __m128i* datas = (__m128i*) datasOrig;
        if constexpr (X > 1) {
            while (num >= 16) {
                DoFetchMany(datas, d0, d1, d2, d3, bins, std::make_index_sequence<8>());
                num -= 16;
                datas += 8;
                bins += 8;
            }
        }

        for (size_t i = 0; i < num; i += 2) {
            __m128i val = datas[0];
            DoFetch2(bins[0 * 64], d0, val);
            DoFetch2(bins[1 * 64], d1, val);
            DoFetch2(bins[2 * 64], d2, val);
            DoFetch2(bins[3 * 64], d3, val);
            datas[0] = val;
            ++datas;
            ++bins;
        }
    }
};

namespace {
struct TMnSseStaticTraits {
    using TLeaf = int;
    using TDocValues128 = ui64[128];
    static TLeaf zeroes[256];
};
}

TMnSseStaticTraits::TLeaf TMnSseStaticTraits::zeroes[256];

namespace {
template <typename Traits>
struct TFetcherBase {
    using TLeaf = typename Traits::TLeaf;
    using TDocValues128 = typename Traits::TDocValues128;

    alignas(0x10) ui8 Bins[4][128];
    alignas(0x10) const TLeaf *Data[4];
    alignas(0x10) TDocValues128 Vals;
    size_t Num = 0;
    size_t Count = 0;

    TFetcherBase(size_t num)
    : Num(num)
    {
#if defined(_msan_enabled_)
    // Bypass use-of-uninitialized-value error
    memset(Bins, 0, sizeof(Bins));
    memset(Data, 0, sizeof(Data));
#endif
    }

    void InitVals(TDocValues128&& vals) {
        Vals = std::move(vals);
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
            Alloc<X>(Traits::zeroes);
        }
        DoFetch<X>();
    }

    template <size_t X>
    NFORCED_INLINE bool DoFetch() {
        if (Count == 4){
            TFetcherWorker<TLeaf, TDocValues128, X>::DoFetch128(Data[0], Data[1], Data[2], Data[3], (const ui16 *)&Bins[0][0], &Vals, Num);
            Count = 0;
        }
        return Count == 0;
    }
};

struct TFetcher : public TFetcherBase<TMnSseStaticTraits> {
    using Base = TFetcherBase<TMnSseStaticTraits>;

    TFetcher(size_t num)
    : Base(num)
    {
        ClearVals();
    }

    inline void ClearVals() {
        Base::Count = 0;
        memset(Base::Vals, 0, sizeof(Base::Vals));
        for (size_t i = 0; i < 4; ++i) {
            Base::Data[i] = TMnSseStaticTraits::zeroes;
        }
    }
};
}

template<size_t Num, typename T, typename Fetcher, size_t Depth>
static void CalculateAllTreesOnDepth(
    const TMnSseStaticMeta& info,
    const typename Fetcher::TLeaf*& data,
    const T*& beginDataIndices,
    const T* endDataIndices,
    Fetcher* fetcher,
    const ui32* val,
    const int treeRangeStart,
    const int treeRangeFinish,
    int& proceed
) {
    const int size = info.GetSizeToCount(Depth);
    const int startIndex = std::max(0, treeRangeStart - proceed);
    const int endIndex = std::min(size, treeRangeFinish - proceed);
    const T* beginDataIndicesCopy = beginDataIndices;
    const typename Fetcher::TLeaf* dataCopy = data;
    beginDataIndicesCopy += Depth * startIndex;
    dataCopy += (1 << Depth) * startIndex;
    for (int i = startIndex; i < endIndex; ++i) {
        DoMXLane<Num, Depth, T>(val, beginDataIndicesCopy, endDataIndices, fetcher->template Alloc<Num>(dataCopy));
        beginDataIndicesCopy += Depth;
        dataCopy += (1 << Depth);
    }
    beginDataIndices += Depth * size;
    data += (1 << Depth) * size;
    proceed += size;
}

template <size_t Num, typename T, typename Fetcher>
static void CalculateAllTrees(
    const TMnSseStaticMeta& info,
    const typename Fetcher::TLeaf* data,
    const T *dataIndices,
    Fetcher* fetcher,
    const ui32* val,
    const int treeRangeStart = 0,
    const int treeRangeFinish = Max<int>()
) {
    const T *indices = dataIndices;
    const T *end = info.DataIndicesSize + indices;
    int proceed = 0;

    CalculateAllTreesOnDepth<Num, T, Fetcher, 0>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 1>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 2>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 3>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 4>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 5>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 6>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 7>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
    CalculateAllTreesOnDepth<Num, T, Fetcher, 8>(info, data, indices, end, fetcher, val, treeRangeStart, treeRangeFinish, proceed);
}

template<size_t Num, typename T>
static void MxNet128ClassicApply(const TPreparedSubBatch& preparedSubBatch, TFetcher* fetcher, const TMnSseStatic &info, const T *dataIndices, size_t rangeBegin, size_t rangeFinish, double* res) {
    const TVector<TMultiData::TLeafData>& multiData = Get<TMultiData>(info.Leaves.Data).MultiData;

    for (size_t dataIndex = 0; dataIndex < multiData.size(); ++dataIndex) {
        ui64 sub = info.Meta.GetSizeToCount(0) +
            info.Meta.GetSizeToCount(1) +
            info.Meta.GetSizeToCount(2) +
            info.Meta.GetSizeToCount(3) +
            info.Meta.GetSizeToCount(4) +
            info.Meta.GetSizeToCount(5) +
            info.Meta.GetSizeToCount(6) +
            info.Meta.GetSizeToCount(7) +
            info.Meta.GetSizeToCount(8);

        double k = 1.0;
        if (!(rangeBegin == 0 && rangeFinish == Max<size_t>())) {
            Y_FAIL("!(rangeBegin == 0 && rangeFinish == Max<size_t>())");
            CalculateAllTrees<Num, T, TFetcher>(info.Meta, multiData[dataIndex].Data, dataIndices,
                fetcher, preparedSubBatch.GetVal(), rangeBegin, rangeFinish);
            k = double(rangeFinish - rangeBegin) / sub;
        } else {
            CalculateAllTrees<Num, T, TFetcher>(info.Meta, multiData[dataIndex].Data, dataIndices,
                fetcher, preparedSubBatch.GetVal());
        }
        fetcher->template DoFlush<Num>();

        sub = (sub << 31);
        const auto multiDataSize = multiData.size();
        const double dataScale = multiData[dataIndex].Norm.DataScale;
        const double dataBias = multiData[dataIndex].Norm.DataBias;
        const i64* src = (i64*)fetcher->Vals;
        if (multiDataSize == 1) {
            for (size_t i = 0; i < preparedSubBatch.NumFactors; ++i) {
                res[i] = (src[i] - sub * k) * dataScale + dataBias * k;
            }
        } else {
            Y_FAIL("multiDataSize != 1");
            double* curResult = &res[dataIndex];
            for (size_t i = 0; i < preparedSubBatch.NumFactors; ++i) {
                *curResult = (src[i] - sub * k) * dataScale + dataBias * k;
                curResult += multiDataSize;
            }
            fetcher->ClearVals();
        }
    }
}

template <typename T>
void MxNetLongApplyTyped(const TMxNetSseBinarization& preparedBatch, const TMnSseStatic &info, double* res, const size_t numValues, size_t rangeBegin, size_t rangeFinish) {
    for (size_t i = 0; i < preparedBatch.Batches.size(); ++i) {
        TFetcher fetcher(preparedBatch.Batches[i].NumFactors);
        switch ((preparedBatch.Batches[i].NumFactors - 1) / 16) {
            case 0:
                MxNet128ClassicApply<1, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
            case 1:
                MxNet128ClassicApply<2, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
            case 2:
                MxNet128ClassicApply<3, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
            case 3:
                MxNet128ClassicApply<4, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
            case 4:
                MxNet128ClassicApply<5, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
            case 5:
                MxNet128ClassicApply<6, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
            case 6:
                MxNet128ClassicApply<7, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
            case 7:
                MxNet128ClassicApply<8, T>(preparedBatch.Batches[i], &fetcher, info, (const T*) info.Meta.DataIndicesPtr, rangeBegin, rangeFinish, res + i * 128 * numValues);
                break;
        }
    }
}

void MxNetLongApply(const TMxNetSseBinarization& preparedBatch, const TMnSseStatic &info, double* res, const size_t numValues, size_t rangeBegin, size_t rangeFinish) {
    if (info.Meta.Has16Bits) {
        Y_FAIL("16Bits");
        return MxNetLongApplyTyped<ui16>(preparedBatch, info, res, numValues, rangeBegin, rangeFinish);
    }
    return MxNetLongApplyTyped<ui32>(preparedBatch, info, res, numValues, rangeBegin, rangeFinish);
}
}

TVector<double> MxNetSseApply(const TMnSseStatic &info, const TMxNetSseBinarization& preparedBatch) {
    TVector<double> res(preparedBatch.GetNumDocs());
    MxNetLongApply(preparedBatch, info, res.data(), 1, 0, Max<size_t>());
    return res;
}
