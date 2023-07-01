/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-19 20:19:24
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-06-19 15:07:27
 * @FilePath:
 * \sph_seepage_flows\seepage_flow\include\vtk_helper\vtk_polygonal_writer.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _VTK_POLYGONAL_WRITER_H_
#define _VTK_POLYGONAL_WRITER_H_

#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolygon.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedLongLongArray.h>
#include <vtkVertex.h>

class VTKPolygonalWriter {
public:
  VTKPolygonalWriter(const std::string path, const std::vector<float3> &p) {
    mPath = path;
    mPointSize = p.size();

    mPolyData = vtkSmartPointer<vtkPolyData>::New();
    mPoints = vtkSmartPointer<vtkPoints>::New();

    CreateStructuredGrid(p);
    mPolyData->SetPoints(mPoints);
  }

  VTKPolygonalWriter(const std::string path, const std::vector<float3> &p,
                     const std::vector<size_t> &labels, const size_t label_id) {
    mPath = path;
    mEnableSplitExporter = true;
    mLabel = labels;
    mLabelId = label_id;
    mPointSize = 0;

    mPolyData = vtkSmartPointer<vtkPolyData>::New();
    mPoints = vtkSmartPointer<vtkPoints>::New();

    CreateStructuredGrid(p);
  }

  void AddIntData(const std::string &name, const std::vector<int> &data) {
    vtkSmartPointer<vtkIntArray> array = vtkSmartPointer<vtkIntArray>::New();
    array->SetName(name.c_str());
    for (auto i = 0; i < data.size(); i++)
      if (!mEnableSplitExporter ||
          (mEnableSplitExporter && mLabel[i] == mLabelId))
        array->InsertNextValue(data[i]);

    mPolyData->GetPointData()->AddArray(array);
  }

  void AddSizeTData(const std::string &name, const std::vector<size_t> &data) {
    vtkSmartPointer<vtkUnsignedLongLongArray> array =
        vtkSmartPointer<vtkUnsignedLongLongArray>::New();
    array->SetName(name.c_str());
    for (auto i = 0; i < data.size(); i++) {
      if (!mEnableSplitExporter ||
          (mEnableSplitExporter && mLabel[i] == mLabelId)) {
        array->InsertNextValue(data[i]);
      }
    }

    mPolyData->GetPointData()->AddArray(array);
  }

  void AddFloatData(const std::string &name, const std::vector<float> &data) {
    vtkSmartPointer<vtkFloatArray> array =
        vtkSmartPointer<vtkFloatArray>::New();
    array->SetName(name.c_str());
    for (auto i = 0; i < data.size(); i++)
      if (!mEnableSplitExporter ||
          (mEnableSplitExporter && mLabel[i] == mLabelId))
        array->InsertNextValue(data[i]);
    mPolyData->GetPointData()->AddArray(array);
  }

  void AddDoubleData(const std::string &name, const std::vector<double> &data) {
    vtkSmartPointer<vtkDoubleArray> array =
        vtkSmartPointer<vtkDoubleArray>::New();
    array->SetName(name.c_str());
    for (auto i = 0; i < data.size(); i++)
      if (!mEnableSplitExporter ||
          (mEnableSplitExporter && mLabel[i] == mLabelId))
        array->InsertNextValue(data[i]);
    mPolyData->GetPointData()->AddArray(array);
  }

  void AddVectorFloatData(const std::string &name,
                          const std::vector<float3> &data) {
    vtkSmartPointer<vtkFloatArray> vectors =
        vtkSmartPointer<vtkFloatArray>::New();
    vectors->SetName(name.c_str());
    vectors->SetNumberOfComponents(3);
    vectors->SetNumberOfTuples(mPointSize);
    int counter = 0;
    for (auto i = 0; i < data.size(); ++i) {
      if (!mEnableSplitExporter ||
          (mEnableSplitExporter && mLabel[i] == mLabelId)) {
        std::vector<float> vec = {data[i].x, data[i].y, data[i].z};
        vectors->SetTuple(counter++, vec.data());
      }
    }

    mPolyData->GetPointData()->AddArray(vectors);
  }

  void WriteToFile(const int version = 42) {
    vtkSmartPointer<vtkPolyDataWriter> writer =
        vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName(mPath.c_str());
    writer->SetFileVersion(version);
    writer->SetInputData(mPolyData);
    // writer->SetFileTypeToBinary();
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

  void CreateStructuredGrid(const std::vector<float3> &p) {

    vtkSmartPointer<vtkCellArray> vertices =
        vtkSmartPointer<vtkCellArray>::New();

    int counter = 0;
    for (auto i = 0; i < p.size(); i++) {
      if (!mEnableSplitExporter ||
          (mEnableSplitExporter && mLabel[i] == mLabelId)) {
        vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();

        if (p[i].x != p[i].x)
          std::cout << "VTK Data: " << std::endl;

        mPoints->InsertNextPoint(p[i].x, p[i].y, p[i].z);
        vertex->GetPointIds()->SetId(0, counter++);
        vertices->InsertNextCell(vertex);
      }
    }

    mPolyData->SetVerts(vertices);
    mPointSize = counter;
    mPolyData->SetPoints(mPoints);
  }
};

typedef std::shared_ptr<VTKPolygonalWriter> VTKPolygonalWriterPtr;

#endif