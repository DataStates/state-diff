#ifndef COMP_FUNC_HPP
#define COMP_FUNC_HPP
#include <Kokkos_Core.hpp>
enum CompareOp {
  Equivalence=0,
  Relative=1,
  Absolute=2
};

template<typename T>
struct BaseComp {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& a, const T& b, double tol) const {
      return true;
    }
};

template<typename T>
struct EquivalenceComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& a, const T& b, double tol) const {
        return a == b;
    }
};

template<typename T>
struct RelativeComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& expect, const T& approx, double tol) const {
        return Kokkos::abs((approx-expect)/expect) <= tol;
    }
};

template<typename T>
struct AbsoluteComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& a, const T& b, double tol) const {
//        float error = static_cast<float>(tol);
//        uint32_t* errorBits = reinterpret_cast<uint32_t*>(&error);
//        int errorExponent = (((*errorBits) >> 23) & 0xFF) - 127;
//        return Kokkos::abs(b-a) <= pow(2, errorExponent);
        return Kokkos::abs(static_cast<double>(b)-static_cast<double>(a)) <= tol;
    }
};
#endif

