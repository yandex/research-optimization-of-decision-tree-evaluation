#pragma once
#include <util/util.h>
namespace NMatrixnet {

/// Feature index with number of borders in matrixnet model.
    struct TFeature {
        /// Feature index.
        ui32 Index = 0;

        /// Number of borders for this factor.
        ui32 Length = 0;

        TFeature() = default;

        TFeature(const ui32 index, const ui32 length)
                : Index(index), Length(length) {
        }
    };

// ----------------------------------------------------------------------------------------------

    struct TNormAttributes {
        double DataBias = 0.0;
        double DataScale = 1.0;

        TNormAttributes() = default;

        TNormAttributes(double dataBias, double dataScale)
                : DataBias(dataBias)
                , DataScale(dataScale) {}
    };


// ----------------------------------------------------------------------------------------------

    struct TMultiData {
        struct TLeafData {
            const int *Data;
            TNormAttributes Norm;

            TLeafData() = default;

            TLeafData(const int *data, double dataBias, double dataScale)
                    : Data(data), Norm(dataBias, dataScale) {}
        };

        TVector <TLeafData> MultiData;
        // Total length of leaf values array, should be the same for all leaf values in model
        size_t DataSize;

        TMultiData() = default;

        TMultiData(const TVector <TLeafData> &multiData, size_t dataSize)
                : MultiData(multiData), DataSize(dataSize) {}

    };

    struct TMultiDataCompact {
    };

    struct TMnSseStaticMeta {
        /// Feature border values array
        const float *Values = nullptr;
        size_t ValuesSize = 0;

        /// List of features, sorted by factor index
        const TFeature *Features = nullptr;
        size_t FeaturesSize = 0;

        /// If Has16Bits == true DataIndicesPtr points to ui16[] array, else it points to ui32[] array
        const void *DataIndicesPtr = nullptr;
        size_t DataIndicesSize = 0;
        bool Has16Bits = true;

        /// Default directions for nan features (NMatrixnetIdl::EFeatureDirection)
        const i8 *MissedValueDirections = nullptr;
        size_t MissedValueDirectionsSize = 0;

        size_t GetIndex(size_t index) const {
            if (Has16Bits) {
                return ((ui16 *) DataIndicesPtr)[index] / 4;
            }
            return ((ui32 *) DataIndicesPtr)[index] / 4;
        }

        const int *SizeToCount = nullptr;
        size_t SizeToCountSize = 0;

        int GetSizeToCount(size_t index) const {
            return index < SizeToCountSize ? SizeToCount[index] : 0;
        }
    };

    struct TMnSseStaticLeaves {
        TVariant <TMultiData, TMultiDataCompact> Data;

        TMnSseStaticLeaves(const TVariant <TMultiData, TMultiDataCompact> &leaves)
                : Data(leaves) {}
    };

    struct TMnSseStatic {
        TMnSseStaticMeta Meta;
        TMnSseStaticLeaves Leaves;

        TMnSseStatic()
                : Meta(), Leaves(TMultiData({}, 0)) {}

        TMnSseStatic(const TMnSseStaticMeta& features, const TMnSseStaticLeaves& leaves)
            : Meta(features), Leaves(leaves)
            {
        }
    };
}