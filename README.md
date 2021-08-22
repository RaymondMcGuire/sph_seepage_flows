# Visual Simulation of Soil-Structure Destruction with Seepage Flows (SCA 2021)

*Xu Wang, Makoto Fujisawa, Masahiko Mikawa

![](./pics/teaser.png)

[[Paper]](https://) [[Video]](https://)

[![WindowsCUDA](https://github.com/RaymondMcGuire/sph_seepage_flows/actions/workflows/WindowsCUDA.yml/badge.svg?branch=main)](https://github.com/RaymondMcGuire/sph_seepage_flows/actions/workflows/WindowsCUDA.yml)

Seepage Flows + WCSPH(CUDA version).

## Environment

- C++ & CUDA10.2
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
- choose your visual studio version(vs15/vs17/vs19)
- run the bat file
