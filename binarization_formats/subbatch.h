#pragma once

#include "../util/util.h"

class TPreparedSubBatch {
public:
    TPreparedSubBatch(size_t numFactors, ui32* buffer)
        : NumFactors(numFactors)
    {
        Val = buffer;
        Y_ASSERT(((uintptr_t)Val & 0xf) == 0);
    }

    ui32* GetVal() {
        return Val;
    }
    const ui32* GetVal() const {
        return Val;
    }

    size_t NumFactors = 0;
private:
    ui32* Val = nullptr;
    TArrayHolder<ui8> Hold;
};
