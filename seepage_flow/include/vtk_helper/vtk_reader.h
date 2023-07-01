/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-19 20:19:24
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-19 21:38:53
 * @FilePath: \sph_seepage_flows\seepage_flow\include\vtk_helper\vtk_reader.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#ifndef _VTK_READER_H_
#define _VTK_READER_H_

#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
class VTKReader {
public:
  static std::vector<float3> ReadPoints(const std::string &filename) {
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();
    vtkSmartPointer<vtkPoints> points = polyData->GetPoints();

    std::vector<float3> result;
    if (points) {
      std::cout << "VTK File: " << filename
                << "; Points Number: " << points->GetNumberOfPoints()
                << std::endl;

      for (auto i = 0; i < points->GetNumberOfPoints(); ++i) {
        double *coords = points->GetPoint(i);
        result.emplace_back(make_float3(coords[0], coords[1], coords[2]));

        if (coords[0] != coords[0] || coords[1] != coords[1] ||
            coords[2] != coords[2])
          std::cout << "VTK Data: " << coords[0] << " " << coords[1] << " "
                    << coords[2] << std::endl;
      }
    } else {
      std::cout << "VTK File: " << filename << "; Cannot Read Point Data!!!"
                << std::endl;
    }

    return result;
  }
};

#endif