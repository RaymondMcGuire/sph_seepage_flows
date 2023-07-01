/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkUniformGridAMR.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   vtkUniformGridAMR
 * @brief   a concrete implementation of vtkCompositeDataSet
 *
 * vtkUniformGridAMR is an AMR (hierarchical) composite dataset that holds vtkUniformGrids.
 *
 * @sa
 * vtkUniformGridAMRDataIterator
 */

#ifndef vtkUniformGridAMR_h
#define vtkUniformGridAMR_h

#include "vtkCommonDataModelModule.h" // For export macro
#include "vtkCompositeDataSet.h"

class vtkCompositeDataIterator;
class vtkUniformGrid;
class vtkAMRInformation;
class vtkAMRDataInternals;

class VTKCOMMONDATAMODEL_EXPORT vtkUniformGridAMR : public vtkCompositeDataSet
{
public:
  static vtkUniformGridAMR* New();
  vtkTypeMacro(vtkUniformGridAMR, vtkCompositeDataSet);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Return a new iterator (the iterator has to be deleted by the user).
   */
  VTK_NEWINSTANCE vtkCompositeDataIterator* NewIterator() override;

  /**
   * Return class name of data type (see vtkType.h for definitions).
   */
  int GetDataObjectType() override { return VTK_UNIFORM_GRID_AMR; }

  /**
   * Restore data object to initial state.
   */
  void Initialize() override;

  /**
   * Initialize the AMR with a specified number of levels and the blocks per level.
   */
  virtual void Initialize(int numLevels, const int* blocksPerLevel);

  /**
   * Set/Get the data description of this uniform grid instance,
   * e.g. VTK_XYZ_GRID
   */
  void SetGridDescription(int gridDescription);
  int GetGridDescription();

  /**
   * Get number of levels.
   */
  unsigned int GetNumberOfLevels();

  /**
   * Get the total number of blocks, including nullptr blocks
   */
  virtual unsigned int GetTotalNumberOfBlocks();

  /**
   * Get the number of datasets at the given level, including null blocks
   */
  unsigned int GetNumberOfDataSets(const unsigned int level);

  ///@{
  /**
   * Get the (min/max) bounds of the AMR domain.
   */
  void GetBounds(double bounds[6]);
  const double* GetBounds();
  void GetMin(double min[3]);
  void GetMax(double max[3]);
  ///@}

  /**
   * Overriding superclass method.
   */
  void SetDataSet(vtkCompositeDataIterator* iter, vtkDataObject* dataObj) override;

  /**
   * At the passed in level, set grid as the idx'th block at that level. idx must be less
   * than the number of data sets at that level
   */
  virtual void SetDataSet(unsigned int level, unsigned int idx, vtkUniformGrid* grid);

  // Needed because, otherwise vtkCompositeData::GetDataSet(unsigned int flatIndex) is hidden.
  using Superclass::GetDataSet;

  /**
   * Get the data set pointed to by iter
   */
  vtkDataObject* GetDataSet(vtkCompositeDataIterator* iter) override;

  /**
   * Get the data set using the (level, index) pair.
   */
  vtkUniformGrid* GetDataSet(unsigned int level, unsigned int idx);

  /**
   * Retrieves the composite index associated with the data at the given
   * (level,index) pair.
   */
  int GetCompositeIndex(const unsigned int level, const unsigned int index);

  /**
   * Given the compositeIdx (as set by SetCompositeIdx) this method returns the
   * corresponding level and dataset index within the level.
   */
  void GetLevelAndIndex(const unsigned int compositeIdx, unsigned int& level, unsigned int& idx);

  ///@{
  /**
   * ShallowCopy.
   */
  void ShallowCopy(vtkDataObject* src) override;
  void RecursiveShallowCopy(vtkDataObject* src) override;
  ///@}

  /**
   * DeepCopy.
   */
  void DeepCopy(vtkDataObject* src) override;

  /**
   * CopyStructure.
   */
  void CopyStructure(vtkCompositeDataSet* src) override;

  ///@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static vtkUniformGridAMR* GetData(vtkInformation* info);
  static vtkUniformGridAMR* GetData(vtkInformationVector* v, int i = 0);
  ///@}

protected:
  vtkUniformGridAMR();
  ~vtkUniformGridAMR() override;

  double Bounds[6];

  /**
   * Get the meta AMR meta data
   */
  vtkGetObjectMacro(AMRData, vtkAMRDataInternals);

  vtkAMRDataInternals* AMRData;

  ///@{
  /**
   * Get/Set the meta AMR meta info
   */
  vtkGetObjectMacro(AMRInfo, vtkAMRInformation);
  virtual void SetAMRInfo(vtkAMRInformation*);
  ///@}

  vtkAMRInformation* AMRInfo;

private:
  vtkUniformGridAMR(const vtkUniformGridAMR&) = delete;
  void operator=(const vtkUniformGridAMR&) = delete;

  friend class vtkUniformGridAMRDataIterator;
};

#endif
