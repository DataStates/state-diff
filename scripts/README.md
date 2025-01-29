# state-diff
Compute differences between immutable data states

## Benchmarking

To test the performance of liburing vs posix and DD, run the command after having compiled the state-diff project

`
bash benchmark.sh local read
`

This command will create a random file of 2GB in you current local directory and run a series of instructions written in *benchmark.sh* and write the throughput and other stats to stdout in addition to a file called *validate_liburing.csv*.