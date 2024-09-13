#include "adios_reader.hpp"

adios_reader_t::adios_reader_t() {}

adios_reader_t::~adios_reader_t() {}

adios_reader_t::adios_reader_t(std::string &name) : fname(name) {}

adios_reader_t::adios_reader_t(const adios_reader_t &other) {
    fname = other.fname;
}

adios_reader_t &
adios_reader_t::operator=(const adios_reader_t &other) {
    if (this == &other) {
        return *this;
    }
    fname = other.fname;
    return *this;
}

size_t
adios_reader_t::read_all() {
    adios2::ADIOS adios_client;
    adios2::IO io = adios_client.DeclareIO("reader");
    io.SetEngine("BP5");
    adios2::Engine reader = io.Open(fname, adios2::Mode::Read);
    reader.BeginStep();
    adios2::Variable<uint8_t> variable = io.InquireVariable<uint8_t>("data");
    reader.Get<uint8_t>(variable, buffer);
    reader.EndStep();
    reader.Close();
    return buffer.size();
}

int
adios_reader_t::enqueue_reads(const std::vector<segment_t> &segments) {
    return 0;
}

int
adios_reader_t::wait(size_t id) {
    return 0;
}

int
adios_reader_t::wait_all() {
    return 0;
}

size_t
adios_reader_t::wait_any() {
    return 0;
}
