/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkTriangleFilter.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   vtkTriangleFilter
 * @brief   convert input polygons and strips to triangles
 *
 * vtkTriangleFilter generates triangles from input polygons and triangle
 * strips.  It also generates line segments from polylines unless PassLines
 * is off, and generates individual vertex cells from vtkVertex point lists
 * unless PassVerts is off.
 */

#ifndef vtkTriangleFilter_h
#define vtkTriangleFilter_h

#include "vtkFiltersCoreModule.h" // For export macro
#include "vtkPolyDataAlgorithm.h"

class VTKFILTERSCORE_EXPORT vtkTriangleFilter : public vtkPolyDataAlgorithm
{
public:
  static vtkTriangleFilter* New();
  vtkTypeMacro(vtkTriangleFilter, vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  ///@{
  /**
   * Turn on/off passing vertices through filter (default: on).
   * If this is on, then the input vertex cells will be broken
   * into individual vertex cells (one point per cell).  If it
   * is off, the input vertex cells will be ignored.
   */
  vtkBooleanMacro(PassVerts, vtkTypeBool);
  vtkSetMacro(PassVerts, vtkTypeBool);
  vtkGetMacro(PassVerts, vtkTypeBool);
  ///@}

  ///@{
  /**
   * Turn on/off passing lines through filter (default: on).
   * If this is on, then the input polylines will be broken
   * into line segments.  If it is off, then the input lines
   * will be ignored and the output will have no lines.
   */
  vtkBooleanMacro(PassLines, vtkTypeBool);
  vtkSetMacro(PassLines, vtkTypeBool);
  vtkGetMacro(PassLines, vtkTypeBool);
  ///@}

  ///@{
  /**
   * Optionally specify the polygon triangulation tolerance to use.  This
   * simply passes the tolerance to the internal vtkPolygon::Tolerance used
   * for triangulation of polygons.  This is for advanced usage, and
   * generally does not need to be set unless tessellation of n-sided
   * polygons with n>4 is required, and special accuracy requirements are
   * needed.  Note that by default if a value <=0 is specified, then the
   * default vtkPolygon::Tolerance is used.
   */
  vtkSetMacro(Tolerance, double);
  vtkGetMacro(Tolerance, double);
  ///@}

protected:
  vtkTriangleFilter()
    : PassVerts(1)
    , PassLines(1)
    , Tolerance(-1.0) // use default vtkPolygon::Tolerance
  {
  }
  ~vtkTriangleFilter() override = default;

  // Usual data generation method
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  vtkTypeBool PassVerts;
  vtkTypeBool PassLines;
  double Tolerance;

private:
  vtkTriangleFilter(const vtkTriangleFilter&) = delete;
  void operator=(const vtkTriangleFilter&) = delete;
};

#endif
