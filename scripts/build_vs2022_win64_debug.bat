cd ..
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --parallel 4 --config Debug --target seepage_flow
pause
