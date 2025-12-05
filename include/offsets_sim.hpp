#ifndef __OFFT_SIM_HPP
#define __OFFT_SIM_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <optional>

enum class Pattern { Sequential, RandomSparse, RandomWithNeighborBias };

class offsets_sim {

    Pattern parse_pattern(const std::string& s) {
        if (s == "seq") return Pattern::Sequential;
        if (s == "rand") return Pattern::RandomSparse;
        if (s == "biased") return Pattern::RandomWithNeighborBias;
        throw std::runtime_error("Unknown pattern: " + s + " (expected: seq|rand|biased)");
    }

public:
    offsets_sim() {};
    ~offsets_sim() {};

    void write_offsets_to_txt(const std::vector<size_t>& offsets,
                                 const std::string& path);

    std::vector<size_t> generate_offsets(size_t num_offsets,
                                            size_t file_size_bytes,
                                            size_t chunk_size_bytes,
                                            Pattern pattern,
                                            double neighbor_pct = 0.0,
                                            size_t neighbor_chunks = 0,
                                            uint64_t seed = 12345);

};

void offsets_sim::write_offsets_to_txt(const std::vector<size_t>& offsets,
                                 const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open output file: " + path);
    for (size_t off : offsets) out << off << '\n';
}

std::vector<size_t> offsets_sim::generate_offsets(size_t num_offsets,
                                            size_t file_size_bytes,
                                            size_t chunk_size_bytes,
                                            Pattern pattern,
                                            double neighbor_pct,     // for biased
                                            size_t neighbor_chunks,    // gap between chunks
                                            uint64_t seed) {

    size_t nchunks = file_size_bytes / chunk_size_bytes;
    if (num_offsets > nchunks)
        throw std::runtime_error("Requested more unique offsets than available chunks");

    std::mt19937_64 rng(seed);
    std::vector<size_t> offsets;
    offsets.reserve(num_offsets);

    switch (pattern) {
        case Pattern::Sequential: {
            // Take the first num_offsets chunks starting from index 0.
            for (size_t i = 0; i < num_offsets; ++i) {
                size_t chunk = i;
                offsets.push_back(chunk * chunk_size_bytes);
            }
            break;
        }

        case Pattern::RandomSparse: {
            // Generate total number of chunks, shuffle and resize
            std::vector<size_t> idx(nchunks);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            for (size_t i = 0; i < num_offsets; ++i) {
                size_t chunk = idx[i];
                offsets.push_back(chunk * chunk_size_bytes);
            }
            break;
        }

        case Pattern::RandomWithNeighborBias: {
            if (neighbor_pct < 0.0 || neighbor_pct > 100.0)
                throw std::runtime_error("neighbor_pct must be in [0, 100]");

            if (num_offsets == 0) return offsets;

            std::unordered_set<size_t> seen;
            seen.reserve(num_offsets * 2);

            // first offset
            std::uniform_int_distribution<size_t> uni(0, nchunks - 1);
            size_t prev = uni(rng);
            seen.insert(prev);
            offsets.push_back(prev * chunk_size_bytes);

            std::bernoulli_distribution choose_neighbor(neighbor_pct / 100.0);

            while (offsets.size() < num_offsets) {
                size_t next_chunk = nchunks; // invalid, but set for later verification

                // to better randomize offts, we determine if we want the next offt to be within gap
                bool try_neighbor = (neighbor_chunks > 0) && choose_neighbor(rng);

                // if we want to next offset to be within gap
                if (try_neighbor) {
                    const size_t lo = (prev > neighbor_chunks) ? (prev - neighbor_chunks) : 0;
                    const size_t hi = std::min(nchunks - 1, prev + neighbor_chunks);

                    // enumerate unseen candidates in the neighbor window (simple, deterministic)
                    std::vector<size_t> candidates;
                    candidates.reserve(hi - lo + 1);
                    for (size_t b = lo; b <= hi; ++b) {
                        if (!seen.count(b)) candidates.push_back(b);
                    }

                    if (!candidates.empty()) {
                        std::uniform_int_distribution<size_t> pick(0, candidates.size() - 1);
                        next_chunk = candidates[pick(rng)];
                    }
                }

                // otherwise or if we were not able to find any valid chunk within gap
                // choose any unseen chunk from the whole space
                if (next_chunk == nchunks) {
                    if (seen.size() >= nchunks) break; // none left

                    // If many remain, repeated random draws are expected to succeed quickly.
                    if (nchunks - seen.size() > 1024) {
                        std::uniform_int_distribution<size_t> pick(0, nchunks - 1);

                        // Note that this loop would only hangs if seen.size() >= nchunks
                        // this scenario was caught earlier
                        while(true) {
                            size_t candidate = pick(rng);
                            if (!seen.count(candidate)) { 
                                next_chunk = candidate; 
                                break; 
                            }
                        }
                    } else {
                        // enumerate remaining and pick first unseen
                        for (size_t b = 0; b < nchunks; ++b) {
                            if (!seen.count(b)) { 
                                next_chunk = b; 
                                break; 
                            }
                        }
                    }
                }

                seen.insert(next_chunk);
                offsets.push_back(next_chunk * chunk_size_bytes);
                prev = next_chunk;
            }

            // This is to guarantee that we were able to supply enough offsets to the user
            if (offsets.size() != num_offsets) {
                throw std::runtime_error("Unable to produce requested number of unique biased offsets "
                                         "(not enough unseen chunks left). Reduce num_offsets or window.");
            }
            break;
        }
    }

    return offsets;
}
#endif