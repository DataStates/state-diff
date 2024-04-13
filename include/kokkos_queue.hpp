#ifndef __KOKKOS_QUEUE_HPP
#define __KOKKOS_QUEUE_HPP
#include <Kokkos_Core.hpp>

/**
 * uint32_t FIFO Queue class. 
 *
 * Simple circular queue with a fixed size using Kokkos for portability
 */
class Queue {
  public:
    Kokkos::View<uint32_t*> queue_d;
    Kokkos::View<uint32_t[1]> len_d;
    Kokkos::View<uint32_t[1]> beg_d;
    Kokkos::View<uint32_t[1]> end_d;
    Kokkos::View<uint32_t*>::HostMirror queue_h;
    Kokkos::View<uint32_t[1]>::HostMirror len_h;
    Kokkos::View<uint32_t[1]>::HostMirror beg_h;
    Kokkos::View<uint32_t[1]>::HostMirror end_h;
    
    /// Constructor
    Queue(uint32_t max_size) {
      queue_d = Kokkos::View<uint32_t*>("Queue", max_size);
      len_d   = Kokkos::View<uint32_t[1]>("Queue length");
      beg_d   = Kokkos::View<uint32_t[1]>("Queue start");
      end_d   = Kokkos::View<uint32_t[1]>("Queue end");
      queue_h = Kokkos::create_mirror_view(queue_d);
      len_h   = Kokkos::create_mirror_view(len_d);
      beg_h   = Kokkos::create_mirror_view(beg_d);
      end_h   = Kokkos::create_mirror_view(end_d);
      Kokkos::deep_copy(len_d, 0);
      Kokkos::deep_copy(beg_d, 0);
      Kokkos::deep_copy(end_d, 0);
    }

    /**
     * Pop first entry from queue and return the value
     */
    KOKKOS_INLINE_FUNCTION uint32_t pop() const {
      uint32_t start = Kokkos::atomic_fetch_add(&beg_d(0), 1);
      start = start % queue_d.extent(0);
//      Kokkos::atomic_decrement(&len_d(0));
      return queue_d(start);
    }

    /**
     * Push entry onto queue
     *
     * \param item    uint32_t value to push onto the queue 
     */
    KOKKOS_INLINE_FUNCTION void push(uint32_t item) const {
      uint32_t end = Kokkos::atomic_fetch_add(&end_d(0), 1);
      end = end % queue_d.extent(0);
//      Kokkos::atomic_increment(&len_d(0));
      queue_d(end) = item;
    }

    /**
     * Push entry onto queue (Call from Host only, syncs with device)
     *
     * \param item    uint32_t value to push onto the queue 
     */
    void host_push(uint32_t item) const {
      uint32_t end = Kokkos::atomic_fetch_add(&end_h(0), 1);
      end = end % queue_h.extent(0);
//      Kokkos::atomic_increment(&len_h(0));
      queue_h(end) = item;
      Kokkos::deep_copy(queue_d, queue_h);
      Kokkos::deep_copy(end_d, end_h);
//      Kokkos::deep_copy(len_d, len_h);
    }

    /**
     * Push entry onto queue without syncing (Call from Host only)
     *
     * \param item    uint32_t value to push onto the queue 
     */
    void host_push_no_sync(uint32_t item) const {
      uint32_t end = Kokkos::atomic_fetch_add(&end_h(0), 1);
      end = end % queue_h.extent(0);
//      Kokkos::atomic_increment(&len_h(0));
      queue_h(end) = item;
    }

    /**
     * Sync host data to device
     */
    void host_to_device() const {
      Kokkos::deep_copy(queue_d, queue_h);
      Kokkos::deep_copy(end_d, end_h);
//      Kokkos::deep_copy(len_d, len_h);
    }

    /**
     * Size of queue (Host only)
     */
    uint32_t size() const {
      Kokkos::deep_copy(beg_h, beg_d);
      Kokkos::deep_copy(end_h, end_d);
      Kokkos::fence();
      if(beg_h(0) <= end_h(0)) {
        return end_h(0) - beg_h(0);
      } else {
        return beg_h(0) + queue_d.extent(0) - end_h(0);
      }
    }

    /**
     * Capacity of queue
     */
    KOKKOS_INLINE_FUNCTION uint32_t capacity() const {
      return queue_d.extent(0);
    }

    /**
     * Fill queue [start,end) (Host only)
     *
     * \param start Start value
     * \param end   End value
     */
    void fill(uint32_t start, uint32_t end) const {
      for(uint32_t i=start; i<end; i++) {
        queue_h(i-start) = i;
      }
      end_h(0) = end-start;
      beg_h(0) = 0;
      Kokkos::deep_copy(beg_d, beg_h);
      Kokkos::deep_copy(end_d, end_h);
      Kokkos::deep_copy(queue_d, queue_h);
    }

    /**
     * Clear queue
     */
    void clear() const {
      Kokkos::deep_copy(queue_d, 0);
      Kokkos::deep_copy(queue_d, 0);
      Kokkos::deep_copy(beg_d, 0);
      Kokkos::deep_copy(end_d, 0);
      Kokkos::deep_copy(beg_h, 0);
      Kokkos::deep_copy(end_h, 0);
    }
};
#endif // KOKKOS_QUEUE_HPP


