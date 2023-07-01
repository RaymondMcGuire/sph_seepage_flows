/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkSMP.h.in

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef vtkSMP_h
#define vtkSMP_h

/* vtkSMPTools back-end */
#define VTK_SMP_Sequential
#define VTK_SMP_BACKEND "Sequential"

// Preprocessor symbols which indicate the availability of backends.
#define VTK_SMP_ENABLE_OPENMP 0
#define VTK_SMP_ENABLE_SEQUENTIAL 1
#define VTK_SMP_ENABLE_STDTHREAD 1
#define VTK_SMP_ENABLE_TBB 0

// Defines which indicate whether the default is a specific backend.
#define VTK_SMP_DEFAULT_IMPLEMENTATION_OPENMP 0
#define VTK_SMP_DEFAULT_IMPLEMENTATION_SEQUENTIAL 1
#define VTK_SMP_DEFAULT_IMPLEMENTATION_STDTHREAD 0
#define VTK_SMP_DEFAULT_IMPLEMENTATION_TBB 0

#endif
