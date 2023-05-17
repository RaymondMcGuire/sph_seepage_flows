/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtk_nlohmannjson.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef vtk_nlohmannjson_h
#define vtk_nlohmannjson_h

/* Use the nlohmannjson library configured for VTK.  */
#define VTK_MODULE_USE_EXTERNAL_vtknlohmannjson 0

#if VTK_MODULE_USE_EXTERNAL_vtknlohmannjson
# define VTK_NLOHMANN_JSON(x) <nlohmann/x>
#else
# define nlohmann vtknlohmann
# define VTK_NLOHMANN_JSON(x) <vtknlohmannjson/include/vtknlohmann/x>
#endif

#endif
