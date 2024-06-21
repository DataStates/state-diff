#ifndef __MODIFIED_KOKKOS_BITSET_HPP
#define __MODIFIED_KOKKOS_BITSET_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_Functional.hpp>

// Assuming necessary Kokkos implementation details are included or accessible

namespace Dedupe {

template<typename Device = Kokkos::DefaultExecutionSpace>
class Bitset;
template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src);

template <typename Device>
class Bitset {
 public:
  using execution_space = typename Device::execution_space;
  using size_type       = unsigned int;

 private:
  enum : unsigned {
    block_size = static_cast<unsigned>(sizeof(unsigned) * CHAR_BIT)
  };
  enum : unsigned { block_mask = block_size - 1u };
  enum : unsigned {
    block_shift = Kokkos::Impl::integral_power_of_two(block_size)
  };

  using block_view_type = Kokkos::View<unsigned*, Device, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  unsigned m_size;
  unsigned m_last_block_mask;
  block_view_type m_blocks;

 public:
  Bitset(unsigned arg_size = 0u)
      : m_size(arg_size), m_last_block_mask(0u) {
    m_blocks = block_view_type(Kokkos::view_alloc(Kokkos::WithoutInitializing, "BitsetBlocks"), (m_size + block_mask) >> block_shift);
    for (int i = 0, end = static_cast<int>(m_size & block_mask); i < end; ++i) {
      m_last_block_mask |= 1u << i;
    }
  }

  KOKKOS_DEFAULTED_FUNCTION
  Bitset(const Bitset<Device>&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  Bitset& operator=(const Bitset<Device>&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  Bitset(Bitset<Device>&&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  Bitset& operator=(Bitset<Device>&&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  ~Bitset() = default;

  KOKKOS_FORCEINLINE_FUNCTION
  unsigned size() const { return m_size; }

  void set() {
    Kokkos::deep_copy(m_blocks, ~0u);
    if (m_last_block_mask) {
      unsigned last_block_val = ~0u & m_last_block_mask;
      Kokkos::deep_copy(Kokkos::subview(m_blocks, m_blocks.extent(0) - 1), last_block_val);
    }
  }

  void reset() { Kokkos::deep_copy(m_blocks, 0u); }

    /// set all bits to 0
  /// can only be called from the host
  void clear() { Kokkos::deep_copy(m_blocks, 0u); }

  KOKKOS_FORCEINLINE_FUNCTION
  bool set(unsigned i) const {
    if (i < m_size) {
      unsigned* block_ptr = &m_blocks(i >> block_shift);
      const unsigned mask = 1u << (i & block_mask);
      return !(Kokkos::atomic_fetch_or(block_ptr, mask) & mask);
    }
    return false;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool reset(unsigned i) const {
    if (i < m_size) {
      unsigned* block_ptr = &m_blocks(i >> block_shift);
      const unsigned mask = 1u << (i & block_mask);
      return Kokkos::atomic_fetch_and(block_ptr, ~mask) & mask;
    }
    return false;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool test(unsigned i) const {
    if (i < m_size) {
      const unsigned block = m_blocks(i >> block_shift);
      const unsigned mask = 1u << (i & block_mask);
      return block & mask;
    }
    return false;
  }

  // New getter method for accessing raw data
  KOKKOS_FORCEINLINE_FUNCTION
  unsigned* data() const {
    return m_blocks.data();
  }

  private:
    template <typename DstDevice, typename SrcDevice>
    friend void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src);
};



template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src) {
  if (dst.size() != src.size()) {
    Kokkos::Impl::throw_runtime_exception(
        "Error: Cannot deep_copy bitsets of different sizes!");
  }

  Kokkos::fence("Bitset::deep_copy: fence before copy operation");
  Kokkos::Impl::DeepCopy<typename DstDevice::memory_space,
                         typename SrcDevice::memory_space>(
      dst.m_blocks.data(), src.m_blocks.data(),
      sizeof(unsigned) * src.m_blocks.extent(0));
  Kokkos::fence("Bitset::deep_copy: fence after copy operation");
}

} // namespace Dedupe

#endif // __MODIFIED_KOKKOS_BITSET_HPP
