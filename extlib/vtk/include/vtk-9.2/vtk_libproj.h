/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtk_libproj.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef vtk_libproj_h
#define vtk_libproj_h

/* Use the libproj library configured for VTK.  */
#define VTK_MODULE_USE_EXTERNAL_vtklibproj 0

#define VTK_LibPROJ_MAJOR_VERSION 8

#if VTK_LibPROJ_MAJOR_VERSION < 5
# define PROJ_VERSION_MAJOR 8
# define proj_list_operations pj_get_list_ref
#endif

#if VTK_MODULE_USE_EXTERNAL_vtklibproj
# if VTK_LibPROJ_MAJOR_VERSION >= 5
#  include <proj.h>
#  include <proj/io.hpp>
#  include <proj/nn.hpp>
#  include <proj/crs.hpp>
# else
#  include <projects.h>
# endif
# include <geodesic.h>
#else
# include <vtklibproj/src/proj.h>
# include <vtklibproj/src/geodesic.h>
# include <vtklibproj/include/proj/io.hpp>
# include <vtklibproj/include/proj/nn.hpp>
# include <vtklibproj/include/proj/crs.hpp>
#endif


#endif
