# RFEPS: Reconstructing Feature-line Equipped Polygonal Surface 
Code of RFEPS.

Thanks for the simple and easy to use BGAL library: https://github.com/BKHao/BGAL

### Dependence

- CGAL 
- Eigen3
- Boost

### Makefile builds (Linux, other Unixes, and Mac. But we recommend using Windows.)

```
git clone https://github.com/Xrvitd/RFEPS
cd RFEPS
mkdir build && cd build
cmake ..
make -j8
make install
```


### MSVC on Windows

```
git clone https://github.com/Xrvitd/RFEPS
```
Open cmake-gui

```
Where is the source code: RFEPS

Where to build the binaries: RFEPS/build
```

note: check the location of dependencies and install. It is recommended to use vcpkg to add dependencies.

Configure->Generate->Open Project

ALL_BUILD->INSTALL



## Test

The example is in 'MAIN'. Include RFEPS in your project when testing and using it.

All the files is in 'RFEPS\data'. 

Please open ``OPENMP`` in Visual Studio to get the best performance.

The Restricted Power Diagram(RPD) in this project is a version that we implemented to facilitate debugging. If you want to get the fastest running speed, please use:
https://github.com/basselin7u/GPU-Restricted-Power-Diagrams


## Testing Platform
- Windows 10 
- Visual Studio 2022
- AMD Ryzen 5950X
- 64GB Momery





Working, a more concise and readable version is coming soon...
