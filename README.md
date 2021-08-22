# sph_seepage_flows

[![WindowsCUDA](https://github.com/RaymondMcGuire/SPH_CUDA/actions/workflows/WindowsCUDA.yml/badge.svg?branch=master)](https://github.com/RaymondMcGuire/SPH_CUDA/actions/workflows/WindowsCUDA.yml)

Seepage Flows + WCSPH(CUDA version).

## Environment

- C++ & CUDA10.2
- Install [CUDA](https://developer.nvidia.com/cuda-downloads) and [Cmake](https://cmake.org/download/) first

## Setup

### Clone Project

```rb
git clone https://github.com/RaymondMcGuire/sph_seepage_flows.git --recursive
```

### Command Line

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

### Scripts

#### For Windows

- cd to ./scripts folder
- choose your visual studio version(vs15/vs17/vs19)
- run the bat file
