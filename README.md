<!--
 * @Author: Xu.WANG
 * @Date: 2021-10-05 21:02:47
 * @LastEditTime: 2022-03-20 17:04:55
 * @LastEditors: Xu.WANG
 * @Description: 
-->
# Visual Simulation of Soil-Structure Destruction with Seepage Flows (SCA 2021)

*Xu Wang, Makoto Fujisawa, Masahiko Mikawa

![](./pics/teaser.png)

[[Paper]](https://raymondmcguire.github.io/seepage_flow/resources/sca2021_preprint.pdf) [[Video]](https://www.youtube.com/embed/zn_mha57URI) [[Project Page]](https://raymondmcguire.github.io/seepage_flow/)

[![WindowsCUDA](https://github.com/RaymondMcGuire/sph_seepage_flows/actions/workflows/WindowsCUDA.yml/badge.svg?branch=main)](https://github.com/RaymondMcGuire/sph_seepage_flows/actions/workflows/WindowsCUDA.yml)

Seepage Flows + WCSPH

## Environment

- C++, CUDA
- Install [CUDA](https://developer.nvidia.com/cuda-downloads) and [Cmake](https://cmake.org/download/) first

## How to run

### Clone Project

```rb
git clone https://github.com/RaymondMcGuire/sph_seepage_flows.git --recursive
```

### Project Initiation (CMake Command Line)

```rb
cd /to/your/project/path
```

```rb
mkdir build
```

```rb
cd build
```

```rb
cmake .. -G"Visual Studio 16 2019" -A x64
```

### Or Scripts

#### For Windows

- cd to ./scripts folder
- choose your visual studio version(vs15/vs17/vs19/vs22)
- run the bat file

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Bibtex

```
@article{10.1145/3480141,
author = {Wang, Xu and Fujisawa, Makoto and Mikawa, Masahiko},
title = {Visual Simulation of Soil-Structure Destruction with Seepage Flows},
year = {2021},
issue_date = {September 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {4},
number = {3},
url = {https://doi.org/10.1145/3480141},
doi = {10.1145/3480141},
journal = {Proc. ACM Comput. Graph. Interact. Tech.},
month = sep,
articleno = {41},
numpages = {18},
keywords = {Seepage Flow, Capillary Action, Dam Breach, Adhesion, Discrete Element Method, Smoothed Particle Hydrodynamics}
}
```
