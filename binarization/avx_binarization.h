#pragma once

#include <binarization_formats/avx_format.h>

TAvxBinarization<512, true> Avx512BinarizationOrdered(const NMatrixnet::TMnSseStaticMeta&, const TVector<TVector<float>>&);

TAvxBinarization<256, true> Avx256BinarizationOrdered(const NMatrixnet::TMnSseStaticMeta&, const TVector<TVector<float>>&);

TAvxBinarization<128, true> Avx128BinarizationOrdered(const NMatrixnet::TMnSseStaticMeta&, const TVector<TVector<float>>&);

TAvxBinarization<512, false> Avx512Binarization(const NMatrixnet::TMnSseStaticMeta&, const TVector<TVector<float>>&);

TAvxBinarization<256, false> Avx256Binarization(const NMatrixnet::TMnSseStaticMeta&, const TVector<TVector<float>>&);

TAvxBinarization<128, false> Avx128Binarization(const NMatrixnet::TMnSseStaticMeta&, const TVector<TVector<float>>&);


TAvxBinarization<512, true> Avx512BinarizationOrderedTransposed(const NMatrixnet::TMnSseStaticMeta&, const TVector<float*>&, size_t numDocs);

TAvxBinarization<256, true> Avx256BinarizationOrderedTransposed(const NMatrixnet::TMnSseStaticMeta&, const TVector<float*>&, size_t numDocs);

TAvxBinarization<128, true> Avx128BinarizationOrderedTransposed(const NMatrixnet::TMnSseStaticMeta&, const TVector<float*>&, size_t numDocs);

TAvxBinarization<512, false> Avx512BinarizationTransposed(const NMatrixnet::TMnSseStaticMeta&, const TVector<float*>&, size_t numDocs);

TAvxBinarization<256, false> Avx256BinarizationTransposed(const NMatrixnet::TMnSseStaticMeta&, const TVector<float*>&, size_t numDocs);

TAvxBinarization<128, false> Avx128BinarizationTransposed(const NMatrixnet::TMnSseStaticMeta&, const TVector<float*>&, size_t numDocs);
