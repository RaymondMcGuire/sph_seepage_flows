/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtk_verdict.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef vtk_verdict_h
#define vtk_verdict_h

/* Use the verdict library configured for VTK.  */
#define VTK_MODULE_USE_EXTERNAL_VTK_verdict 0

#if VTK_MODULE_USE_EXTERNAL_VTK_verdict
# include <verdict.h>
#else
# include <vtkverdict/verdict.h>
#endif

#endif
