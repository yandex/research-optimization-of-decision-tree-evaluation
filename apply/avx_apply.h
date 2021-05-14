#pragma once

#include <binarization_formats/avx_format.h>

TVector<double> Avx128Apply(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, false> &preparedBatch);

TVector<double> Avx256Apply(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, false> &preparedBatch);

TVector<double> Avx512Apply(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch);


TVector<double> Avx128ApplyOrdered(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, true> &preparedBatch);

TVector<double> Avx256ApplyOrdered(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, true> &preparedBatch);

TVector<double> Avx512ApplyOrdered(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, true> &preparedBatch);

TVector<double> Avx128ApplyOneTreeFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, false> &preparedBatch);

TVector<double> Avx256ApplyOneTreeFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, false> &preparedBatch);

TVector<double> Avx512ApplyOneTreeFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch);

TVector<double> Avx128ApplyInstantFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<128, false> &preparedBatch);

TVector<double> Avx256ApplyInstantFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<256, false> &preparedBatch);

TVector<double> Avx512ApplyInstantFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch);

TVector<double> Avx512ApplyShuffleFetch(const NMatrixnet::TMnSseStatic &info, const TAvxBinarization<512, false> &preparedBatch);
