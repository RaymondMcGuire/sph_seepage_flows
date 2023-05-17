/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkOpenGLHyperTreeGridMapper.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   vtkOpenGLHyperTreeGridMapper
 * @brief   map vtkHyperTreeGrid to graphics primitives
 *
 * vtkOpenGLHyperTreeGridMapper is a class that uses OpenGL to do the actual
 * rendering of Hyper Tree Grid.
 */

#ifndef vtkOpenGLHyperTreeGridMapper_h
#define vtkOpenGLHyperTreeGridMapper_h

#include "vtkHyperTreeGridMapper.h"
#include "vtkSetGet.h"       // Get macro
#include "vtkSmartPointer.h" // For vtkSmartPointer

#include "vtkRenderingOpenGL2Module.h" // For export macro

class VTKRENDERINGOPENGL2_EXPORT vtkOpenGLHyperTreeGridMapper : public vtkHyperTreeGridMapper
{
public:
  static vtkOpenGLHyperTreeGridMapper* New();
  vtkTypeMacro(vtkOpenGLHyperTreeGridMapper, vtkHyperTreeGridMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
  vtkOpenGLHyperTreeGridMapper();
  virtual ~vtkOpenGLHyperTreeGridMapper() override = default;

private:
  vtkOpenGLHyperTreeGridMapper(const vtkOpenGLHyperTreeGridMapper&) = delete;
  void operator=(const vtkOpenGLHyperTreeGridMapper&) = delete;
};

#endif
