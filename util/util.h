#pragma once
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <variant>
#include <vector>

typedef int8_t i8;
typedef int16_t i16;
typedef uint8_t ui8;
typedef uint16_t ui16;
typedef uint32_t ui32;
typedef int32_t i32;
typedef uint64_t ui64;
typedef int64_t i64;

#define Y_FAIL(msg) std::abort()

#define Y_PREFETCH_READ(Pointer, Priority) __builtin_prefetch((const void*)(Pointer), 0, Priority)

#define Y_ENSURE assert

#define Y_ASSERT assert

#define NFORCED_INLINE __attribute__((__always_inline__))

#define Min std::min

#define Cerr std::cerr

#define Y_UNUSED(var) (void)(var)

template<typename T>
using TVector = std::vector<T>;

template<typename T, typename U>
using TMap = std::map<T, U>;

using TString = std::string;

template<typename T, typename U>
using TVariant = std::variant<T, U>;

template<typename V, typename T, typename U>
auto& Get(std::variant<T, U>& var) {
    return std::get<V>(var);
}

template<typename V, typename T, typename U>
const auto& Get(const std::variant<T, U>& var) {
    return std::get<V>(var);
}

template<typename T>
static constexpr T Max() noexcept {
    return std::numeric_limits<T>::max();
}

template<typename T>
using TArrayHolder = std::unique_ptr<T[]>;

struct TRandomGenerator {
    explicit TRandomGenerator(int seed) : generator(seed) {}
    float GenRandReal1() {
        return std::uniform_real_distribution<float>()(generator);
    }
    ui64 GenRand64() {
        return std::uniform_int_distribution<ui64>()(generator);
    }
    std::default_random_engine generator;
};