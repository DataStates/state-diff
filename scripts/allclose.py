import argparse
import os
import time
import numpy as np

DTYPE = np.float32
ITEMSIZE = np.dtype(DTYPE).itemsize  # 4 for float32

def read_binary_file_in_chunks(file_path, buffer_bytes, dtype=DTYPE):
    """Yield dtype arrays from file_path in chunks of buffer_bytes (rounded to dtype)."""
    # Round buffer_bytes down to a multiple of dtype size so frombuffer is happy.
    buf = (buffer_bytes // ITEMSIZE) * ITEMSIZE
    if buf == 0:
        raise ValueError(f"buffer_bytes too small for dtype size {ITEMSIZE}")
    with open(file_path, 'rb') as f:
        while True:
            block = f.read(buf)
            if not block:
                break
            # If final block isn't a multiple of dtype size, drop trailing bytes.
            usable = (len(block) // ITEMSIZE) * ITEMSIZE
            if usable == 0:
                break
            yield np.frombuffer(block[:usable], dtype=dtype, count=usable // ITEMSIZE)

def isclose_files(file1, file2, buffer_bytes, rtol=1e-5, atol=1e-8):
    """Compare two binary float32 files chunk-by-chunk using np.isclose."""
    total_mismatch = 0
    total_compare_time = 0.0
    total_elements = 0

    blocks1 = read_binary_file_in_chunks(file1, buffer_bytes)
    blocks2 = read_binary_file_in_chunks(file2, buffer_bytes)

    for a, b in zip(blocks1, blocks2):
        # Truncate to the shortest block to avoid shape mismatch
        n = min(a.size, b.size)
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        t0 = time.time()
        mism = np.count_nonzero(~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False))
        total_compare_time += time.time() - t0

        total_mismatch += int(mism)
        total_elements += int(n)

    # If files differ in length (in elements), count the remainder as mismatches
    sz1 = os.path.getsize(file1) // ITEMSIZE
    sz2 = os.path.getsize(file2) // ITEMSIZE
    if sz1 != sz2:
        total_mismatch += abs(sz1 - sz2)
        total_elements += abs(sz1 - sz2)

    are_close = (total_mismatch == 0)
    return are_close, total_mismatch, total_elements, total_compare_time

def process(file1, file2, buffer_bytes, error, use_relative):
    """Driver that chooses rtol/atol per error type and prints summary."""
    t0 = time.time()
    if use_relative:
        rtol, atol = error, 0.0
    else:
        rtol, atol = 0.0, error

    are_close, mismatches, total_elems, compare_time = isclose_files(
        file1, file2, buffer_bytes, rtol=rtol, atol=atol
    )
    total_time = time.time() - t0

    print(f"Comparing:\n  {file1}\n  {file2}")
    print(f"The arrays are close: {are_close}")
    print(f"Elements compared:    {total_elems}")
    print(f"Mismatched elements:  {mismatches}")
    print(f"Compare time:         {compare_time:.6f} s")
    print(f"Total time (I/O+cmp): {total_time:.6f} s")

    io_time = total_time - compare_time
    return are_close, mismatches, io_time, compare_time

def main():
    p = argparse.ArgumentParser(prog='allclose_cpu')
    p.add_argument('--run0', nargs='+', required=True, help='List of files from run0')
    p.add_argument('--run1', nargs='+', required=True, help='List of files from run1 (same count/order)')
    p.add_argument('--error', type=float, default=1e-3, help='Tolerance value')
    p.add_argument('--buffer', type=int, default=64, help='Buffer size in MiB for streaming I/O')
    p.add_argument('--errortype', choices=['absolute', 'relative'], default='absolute',
                  help='Use absolute or relative error')
    p.add_argument('--log', default='allclose_log.csv', help='CSV log filename')
    args = p.parse_args()

    if len(args.run0) != len(args.run1):
        raise ValueError("run0 and run1 must have the same number of files, in matching order.")

    buffer_bytes = args.buffer * 1024 * 1024
    use_relative = (args.errortype == 'relative')

    # Prepare CSV
    header = "File,Data filesize,Error tolerance,Block size,Elements different,Num comparisons,Compute time,Total time\n"
    log_fname = f"{args.log}.csv"
    write_header = not os.path.exists(log_fname)
    with open(log_fname, 'a' if not write_header else 'w') as outf:
        if write_header:
            outf.write(header)

        for f0, f1 in zip(args.run0, args.run1):
            are_close, mism, io_time, cmp_time = process(f0, f1, buffer_bytes, args.error, use_relative)
            data_size = os.path.getsize(f0)  # bytes
            num_elements = data_size // ITEMSIZE
            outf.write(
                f"{f1},{data_size},{args.error},{buffer_bytes},{mism},{num_elements},{cmp_time:.6f},{(io_time+cmp_time):.6f}\n"
            )

if __name__ == "__main__":
    main()