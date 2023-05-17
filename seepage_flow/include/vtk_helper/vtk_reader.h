
#ifndef _VTK_READER_H_
#define _VTK_READER_H_

#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
class VTKReader {
public:
    static std::vector<float3> ReadPoints(const std::string& filename) {
        vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();

        vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();
        vtkSmartPointer<vtkPoints> points = polyData->GetPoints();

        std::vector<float3> result;
        if (points) {
            std::cout << "VTK File: " << filename<< "; Points Number: "<< points->GetNumberOfPoints() << std::endl;

            for (auto i = 0; i < points->GetNumberOfPoints(); ++i) {
                double* coords = points->GetPoint(i);
                result.emplace_back(make_float3(coords[0],coords[1],coords[2]));
            }
        } else {
            std::cout << "VTK File: " << filename<< "; Cannot Read Point Data!!!" << std::endl;
        }


        return result;
    }
};

#endif