#ifndef __BOOST_ARCHIVE_HPP
#define __BOOST_ARCHIVE_HPP

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace state_diff::boost {
    using namespace ::boost;

    // Serialize multiple objects
    template <typename T1, typename T2>
    std::function<void(std::ostream &)>
    serializer(T1 &info, T2 &tree_obj) {
        return [&info, &tree_obj](std::ostream &out) {
            archive::binary_oarchive ar(out);
            ar << info;
            ar << tree_obj;
        };
    }

    // Deserialize multiple objects
    template <typename T1, typename T2>
    std::function<bool(std::istream &)>
    deserializer(T1 &info, T2 &tree_obj) {
        return [&info, &tree_obj](std::istream &in) {
            try {
                archive::binary_iarchive ar(in);
                ar >> info;
                ar >> tree_obj;
            } catch (std::exception &e) {
                return false;
            }
            return true;
        };
    }
}   // namespace state_diff::boost

#endif   //__BOOST_ARCHIVE_HPP
