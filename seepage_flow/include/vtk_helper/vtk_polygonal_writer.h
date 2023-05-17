/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-16 21:40:10
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 16:42:00
 * @FilePath: \sph_seepage_flows\seepage_flow\include\vtk_helper\vtk_polygonal_writer.h
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#ifndef _VTK_POLYGONAL_WRITER_H_
#define _VTK_POLYGONAL_WRITER_H_

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkDataArray.h>
#include <vtkCellArray.h>
#include<vtkPolygon.h>
#include <vtkPointData.h>
#include <vtkPolyDataWriter.h>
#include<vtkDoubleArray.h>
#include<vtkFloatArray.h>
#include<vtkIntArray.h>
#include<vtkCell.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>

class VTKPolygonalWriter
{
public:
    VTKPolygonalWriter(const std::string path, const std::vector<float3>& p)
    {
        mPath = path;
        mPointSize = p.size();

        mPolyData = vtkSmartPointer<vtkPolyData>::New();
        mPoints = vtkSmartPointer<vtkPoints>::New();

        CreateStructuredGrid(p);
        mPolyData->SetPoints(mPoints);

    }

    VTKPolygonalWriter(const std::string path, const std::vector<float3>& p, const std::vector<size_t>& labels, const size_t label_id)
    {
        mPath = path;
        mEnableSplitExporter = true;
        mLabel = labels;
        mLabelId = label_id;
        mPointSize = 0;

        mPolyData = vtkSmartPointer<vtkPolyData>::New();
        mPoints = vtkSmartPointer<vtkPoints>::New();

        CreateStructuredGrid(p);
        mPolyData->SetPoints(mPoints);

    }

        void AddIntData(const std::string& name, const std::vector<int>& data)
    {
        vtkSmartPointer<vtkIntArray> array = vtkSmartPointer<vtkIntArray>::New();
        array->SetName(name.c_str());
        for (auto i = 0; i < data.size(); i++)
            if(!mEnableSplitExporter || (mEnableSplitExporter && mLabel[i] == mLabelId))
                array->InsertNextValue(data[i]);
        
        mPolyData->GetPointData()->AddArray(array);
    }

    void AddSizeTData(const std::string& name, const std::vector<size_t>& data)
    {
        vtkSmartPointer<vtkIntArray> array = vtkSmartPointer<vtkIntArray>::New();
        array->SetName(name.c_str());
        for (auto i = 0; i < data.size(); i++)
            if(!mEnableSplitExporter || (mEnableSplitExporter && mLabel[i] == mLabelId))
                array->InsertNextValue(data[i]);
        
        mPolyData->GetPointData()->AddArray(array);
    }

        void AddFloatData(const std::string& name, const std::vector<float>& data)
    {
        vtkSmartPointer<vtkFloatArray> array = vtkSmartPointer<vtkFloatArray>::New();
        array->SetName(name.c_str());
        for (auto i = 0; i < data.size(); i++)
            if(!mEnableSplitExporter || (mEnableSplitExporter && mLabel[i] == mLabelId))
                array->InsertNextValue(data[i]);
        mPolyData->GetPointData()->AddArray(array);
    }

    void AddDoubleData(const std::string& name, const std::vector<double>& data)
    {
        vtkSmartPointer<vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();
        array->SetName(name.c_str());
        for (auto i = 0; i < data.size(); i++)
            if(!mEnableSplitExporter || (mEnableSplitExporter && mLabel[i] == mLabelId))
                array->InsertNextValue(data[i]);
        mPolyData->GetPointData()->AddArray(array);
    }

    void AddVectorFloatData(const std::string& name, const std::vector<float3>& data)
    {
    vtkSmartPointer<vtkFloatArray> vectors = vtkSmartPointer<vtkFloatArray>::New();
    vectors->SetName(name.c_str());
    vectors->SetNumberOfComponents(3);
    vectors->SetNumberOfTuples(mPointSize);
     int counter = 0;
    for (auto i = 0; i < data.size(); ++i) {
         if(!mEnableSplitExporter || (mEnableSplitExporter && mLabel[i] == mLabelId))
         {
                    std::vector<float> vec = { data[i].x, data[i].y, data[i].z };
                    vectors->SetTuple(counter++, vec.data());
         }

    }
        

    mPolyData->GetPointData()->AddArray(vectors);

    }

    void WriteToFile(const int version=42)
    {
        vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
        writer->SetFileName(mPath.c_str());
        writer->SetFileVersion(version);
        writer->SetInputData(mPolyData);
        writer->SetFileTypeToBinary();
        writer->Write();
    }

private:
    std::string mPath;
    size_t mPointSize;

    bool mEnableSplitExporter = false;
    std::vector<size_t> mLabel;
    size_t mLabelId;

    vtkSmartPointer<vtkPolyData> mPolyData;
    vtkSmartPointer<vtkPoints> mPoints;

    void CreateStructuredGrid(const std::vector<float3>& p)
    {

        vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();
        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();

        int counter = 0;
        for (auto i = 0; i < p.size(); i++)
        {
            if(!mEnableSplitExporter || (mEnableSplitExporter && mLabel[i] == mLabelId))
            {
                mPoints->InsertNextPoint(p[i].x, p[i].y, p[i].z);
                 polygon->GetPointIds()->InsertNextId(counter++);
            }

        }
            polygons->InsertNextCell(polygon);

        mPolyData->SetPolys(polygons);
        mPointSize = counter;
    }
};

typedef std::shared_ptr<VTKPolygonalWriter> VTKPolygonalWriterPtr;

#endif