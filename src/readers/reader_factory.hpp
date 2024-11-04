#ifndef __READER_FACTORY_HPP
#define __READER_FACTORY_HPP

#include "io_reader.hpp"
#include "liburing_reader.hpp"
#include "posix_reader.hpp"

// Use WITH_LIBURING as default
#if !defined(WITH_ADIOS) && !defined(WITH_MMAP)
#define WITH_LIBURING
#endif

template <typename DataType>
class IOModule
{
public:
    static io_reader_t<DataType> *createReader(size_t buff_len, const std::string &file_name,
                                     const size_t chunk_size, bool async_memcpy = true,
                                     bool transfer_all = true, int nthreads = 2)
    {
#ifdef WITH_LIBURING
        return new posix_reader_t<DataType>(buff_len, file_name, chunk_size,
                                  async_memcpy, transfer_all, nthreads);
//        return new liburing_reader_t<DataType>(buff_len, file_name, chunk_size,
//                                     async_memcpy, transfer_all, nthreads);
#elif defined(WITH_MMAP)
        return new posix_reader_t<DataType>(buff_len, file_name, chunk_size,
                                  async_memcpy, transfer_all, nthreads);
#endif
    }
};

#endif // __READER_FACTORY_HPP
