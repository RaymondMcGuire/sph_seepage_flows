#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkDataArray.h>
#include <vtkCellArray.h>
#include<vtkPolygon.h>
#include <vtkPointData.h>
#include <vtkPolyDataWriter.h>
#include<vtkDoubleArray.h>
#include<vtkCell.h>

class PolygonalMeshData
{
public:
    PolygonalMeshData(int dimX, int dimY, int dimZ, double spacing)
        : dimX_(dimX), dimY_(dimY), dimZ_(dimZ), spacing_(spacing)
    {
        polyData_ = vtkSmartPointer<vtkPolyData>::New();
        points_ = vtkSmartPointer<vtkPoints>::New();

                CreateStructuredGrid();
        polyData_->SetPoints(points_);


    }

    void AddScalarData(const std::string& name, const std::vector<double>& data)
    {
        vtkSmartPointer<vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();
        array->SetName(name.c_str());
        for (double value : data) {
            array->InsertNextValue(value);
        }
        polyData_->GetPointData()->AddArray(array);
    }

    void AddPolygon(const std::vector<vtkIdType>& pointIds)
    {
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
    for (int i = 0; i < pointIds.size(); ++i) {
        polygon->GetPointIds()->InsertNextId(pointIds[i]);
    }
    polygons->InsertNextCell(polygon);

     polyData_->SetPolys(polygons);

    }

    void WriteToFile(const std::string& fileName, const int version=42)
    {
        vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
        writer->SetFileName(fileName.c_str());
        writer->SetFileVersion(version);
        writer->SetInputData(polyData_);
        writer->Write();
    }

private:
    void CreateStructuredGrid()
    {
        for (int k = 0; k < dimZ_; ++k) {
            for (int j = 0; j < dimY_; ++j) {
                for (int i = 0; i < dimX_; ++i) {
                    double x = i * spacing_;
                    double y = j * spacing_;
                    double z = k * spacing_;
                    points_->InsertNextPoint(x, y, z);
                }
            }
        }
    }

    int dimX_;
    int dimY_;
    int dimZ_;
    double spacing_;
    vtkSmartPointer<vtkPolyData> polyData_;
    vtkSmartPointer<vtkPoints> points_;
};