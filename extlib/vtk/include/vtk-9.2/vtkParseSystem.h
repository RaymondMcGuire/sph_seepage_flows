/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkParseSystem.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
  This file contains routines for accessing the file system.
*/

#ifndef vtkParseSystem_h
#define vtkParseSystem_h

#include "vtkParseString.h"
#include "vtkWrappingToolsModule.h"

#include <stdio.h> /* for FILE* */

/**
 * Contains the paths to all files that have been discoved on the file
 * system. This is used to accelerate searches for header files.
 */
typedef struct SystemInfo_
{
  StringCache* Strings;        /* to accelerate string allocation */
  const char*** FileHashTable; /* paths to all discovered files */
  const char*** DirHashTable;  /* paths to all catalogued directories */
} SystemInfo;

/**
 * An enum to identify the types of discovered files
 */
typedef enum system_filetype_t_
{
  VTK_PARSE_NOFILE = 0,
  VTK_PARSE_ISFILE = 1,
  VTK_PARSE_ISDIR = 2
} system_filetype_t;

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Check if a file with the given name exists and return its type:
   * VTK_PARSE_ISDIR, VTK_PARSE_ISFILE, or VTK_PARSE_NOFILE if not found.
   * This will cache results for the entire parent directory in order
   * to accelerate future searches.
   */
  VTKWRAPPINGTOOLS_EXPORT
  system_filetype_t vtkParse_FileExists(SystemInfo* info, const char* name);

  /**
   * Free the memory that the SystemInfo uses to cache the files.
   */
  VTKWRAPPINGTOOLS_EXPORT
  void vtkParse_FreeFileCache(SystemInfo* info);

  /**
   * On Win32, this interpretes fname as UTF8 and then calls wfopen().
   * The returned handle must be freed with fclose().
   */
  VTKWRAPPINGTOOLS_EXPORT
  FILE* vtkParse_FileOpen(const char* fname, const char* mode);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
/* VTK-HeaderTest-Exclude: vtkParseSystem.h */
