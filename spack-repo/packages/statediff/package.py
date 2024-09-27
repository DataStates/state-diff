from spack.package import CMakePackage

class Statediff(CMakePackage):
    """state-diff is a project that depends on Kokkos, liburing, and cereal."""
    homepage = "https://github.com/DataStates/state-diff"
    git      = "https://github.com/DataStates/state-diff.git"

    version('rev', branch='rev-cereal') 

    variant(
        "build_type",
        default="Release",
        description="CMake build type",
        values=("Debug", "Release", "RelWithDebInfo"),
    )

    build_directory = "build"

    # Specify the dependencies
    depends_on('kokkos+openmp')
    depends_on('cmake', type="build")

    def cmake_args(self):
        args = [
            self.define('CMAKE_BUILD_TYPE', self.spec.variants['build_type'].value),
            self.define('CMAKE_CXX_COMPILER', self.spec["kokkos"].kokkos_cxx),
            self.define('CMAKE_CXX_FLAGS', '-fPIC'),
            self.define('CMAKE_INSTALL_PREFIX', self.prefix+'/build'),
            self.define('ENABLE_TESTS', 'ON'),
            self.define('Kokkos_DIR', self.spec['kokkos'].prefix.lib64.cmake.Kokkos)
        ]
        return args

    def install(self, spec, prefix):
        with working_dir(self.build_directory):
            make()
            make('install')
