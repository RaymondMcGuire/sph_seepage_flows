/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCGNSReader.h

  Copyright (c) Ken Martin, Will Schrodeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

  =========================================================================*/
// Copyright 2013-2014 Mickael Philit.

/**
 * @class   vtkCGNSReader
 *
 * vtkCGNSReader creates a multi-block dataset and reads unstructured grids
 * and structured meshes from binary files stored in CGNS file format,
 * with data stored at the nodes, cells or faces.
 *
 * vtkCGNSReader is inspired by the VisIt CGNS reader originally written by
 * B. Whitlock. vtkCGNSReader relies on the low level CGNS API to load data sets
 * and reduce memory footprint.
 */

#ifndef vtkCGNSReader_h
#define vtkCGNSReader_h

#include "vtkIOCGNSReaderModule.h" // for export macro
#include "vtkMultiBlockDataSetAlgorithm.h"
#include "vtkNew.h" // for vtkNew.

#include <string> // for std::string

class vtkDataArraySelection;
class vtkInformationStringKey;

namespace CGNSRead
{
class vtkCGNSMetaData;
}

class vtkMultiProcessController;
class VTKIOCGNSREADER_EXPORT vtkCGNSReader : public vtkMultiBlockDataSetAlgorithm
{
public:
  static vtkCGNSReader* New();
  vtkTypeMacro(vtkCGNSReader, vtkMultiBlockDataSetAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  enum DataArrayLocation
  {
    CELL_DATA = 0,
    FACE_DATA
  };

  ///@{
  /**
   * Set/get the location of the data arrays to read. Possible values for the data location are:
   * - CELL_DATA - Read CellCenter data arrays from the CGNS file. Construct 3D meshes with 3D cells
   * (e.g. a cube is defined as 1 cell).
   * - FACE_DATA - Read FaceCenter data arrays from the CGNS file. Construct 3D meshes with 2D
   * cells/faces (e.g. a cube is defined as 6 quad cells). Element connectivity must be defined with
   * element type NGON_n.
   *
   * Default is CELL_DATA.
   */
  vtkSetClampMacro(DataLocation, int, vtkCGNSReader::CELL_DATA, vtkCGNSReader::FACE_DATA);
  vtkGetMacro(DataLocation, int);
  ///@}

  ///@{
  /**
   * Specify file name of CGNS datafile to read
   */
  vtkSetStdStringFromCharMacro(FileName);
  vtkGetCharFromStdStringMacro(FileName);
  ///@}

  /**
   * Is the given file name a CGNS file?
   */
  int CanReadFile(VTK_FILEPATH const char* filename);

  /**
   * Returns access to the base selection object.
   */
  vtkDataArraySelection* GetBaseSelection();

  /**
   * Returns access to the family selection object.
   */
  vtkDataArraySelection* GetFamilySelection();

  ///@{
  /**
   * API to select bases to read. These calls simply forward to the
   * vtkDataArraySelection instance obtained from `GetBaseSelection()`.
   *
   * By default, 0-th base is enabled and all others are disabled.
   */
  int GetBaseArrayStatus(const char* name);
  void SetBaseArrayStatus(const char* name, int status);
  void DisableAllBases();
  void EnableAllBases();
  int GetNumberOfBaseArrays();
  const char* GetBaseArrayName(int index);
  ///@}

  ///@{
  /**
   * API to select families to read. These calls simply forward to the
   * vtkDataArraySelection instance obtained from `GetFamilySelection()`.
   */
  int GetNumberOfFamilyArrays();
  const char* GetFamilyArrayName(int index);
  void SetFamilyArrayStatus(const char* name, int status);
  int GetFamilyArrayStatus(const char* name);
  void EnableAllFamilies();
  void DisableAllFamilies();
  ///@}

  ///@{
  /**
   * API to get information of point arrays and enable/disable loading of
   * a particular arrays.
   */
  int GetNumberOfPointArrays();
  const char* GetPointArrayName(int index);
  int GetPointArrayStatus(const char* name);
  void SetPointArrayStatus(const char* name, int status);
  void DisableAllPointArrays();
  void EnableAllPointArrays();
  ///@}

  ///@{
  /**
   * API to get information of cell arrays and enable/disable loading of
   * a particular arrays.
   */
  int GetNumberOfCellArrays();
  const char* GetCellArrayName(int index);
  int GetCellArrayStatus(const char* name);
  void SetCellArrayStatus(const char* name, int status);
  void DisableAllCellArrays();
  void EnableAllCellArrays();
  ///@}

  ///@{
  /**
   * API to get information of face arrays and enable/disable loading of
   * a particular arrays.
   */
  int GetNumberOfFaceArrays();
  const char* GetFaceArrayName(int index);
  int GetFaceArrayStatus(const char* name);
  void SetFaceArrayStatus(const char* name, int status);
  void DisableAllFaceArrays();
  void EnableAllFaceArrays();
  ///@}

  vtkSetMacro(DoublePrecisionMesh, int);
  vtkGetMacro(DoublePrecisionMesh, int);
  vtkBooleanMacro(DoublePrecisionMesh, int);

  ///@{
  /**
   * Enable/disable loading of boundary condition patches.
   * Defaults to false.
   */
  vtkSetMacro(LoadBndPatch, bool);
  vtkGetMacro(LoadBndPatch, bool);
  vtkBooleanMacro(LoadBndPatch, bool);
  ///@}

  ///@{
  /**
   * Enable/disable loading of zone mesh. Defaults to true. It may be turned off
   * to load only boundary patches (when LoadBndPatch if ON), for example.
   */
  vtkSetMacro(LoadMesh, bool);
  vtkGetMacro(LoadMesh, bool);
  vtkBooleanMacro(LoadMesh, bool);
  ///@}

  ///@{
  /**
   * Enable/disable adding an empty physical dimension to vectors in case of 2D solutions.
   */
  vtkSetMacro(Use3DVector, bool);
  vtkGetMacro(Use3DVector, bool);
  vtkBooleanMacro(Use3DVector, bool);
  ///@}

  /**
   * This option is provided for debugging and should not be used for production
   * runs as the output data produced may not be correct. When set to true, the
   * read will simply read each solution (`FlowSolution_t`) node encountered in
   * a zone and create a separate block under the block corresponding to the
   * zone in the output.
   */
  vtkSetMacro(CreateEachSolutionAsBlock, int);
  vtkGetMacro(CreateEachSolutionAsBlock, int);
  vtkBooleanMacro(CreateEachSolutionAsBlock, int);

  /**
   * When set to true (default is false), the reader will simply
   * ignore `FlowSolutionPointers` since they are either incomplete or invalid
   * and instead will rely on FlowSolution_t nodes being labelled as
   * "...AtStep<tsindex>" to locate solution nodes for a specific timestep.
   * Note, tsindex starts with 1 (not zero).
   *
   * When set to false, the reader will still try to confirm that at least one
   * valid FlowSolution_t node is referred to in FlowSolutionPointers nodes for the
   * current timestep. If none is found, then the reader will print out a
   * warning and act as if IgnoreFlowSolutionPointers was set to true. To avoid
   * this warning, one should set IgnoreFlowSolutionPointers to true.
   */
  vtkSetMacro(IgnoreFlowSolutionPointers, bool);
  vtkGetMacro(IgnoreFlowSolutionPointers, bool);
  vtkBooleanMacro(IgnoreFlowSolutionPointers, bool);

  /**
   * When set to true (default is false), the reader will try
   * to determine to determine FlowSolution_t nodes to read with a pattern matching
   * This can be useful for unsteady solutions when `FlowSolutionPointers` are not
   * reliable
   */
  vtkSetMacro(UseUnsteadyPattern, bool);
  vtkGetMacro(UseUnsteadyPattern, bool);
  vtkBooleanMacro(UseUnsteadyPattern, bool);

  /**
   * This reader can support piece requests by distributing each block in each
   * zone across ranks (default). To make the reader disregard piece request and
   * read all blocks in the zone, set this to false (default is true).
   */
  vtkSetMacro(DistributeBlocks, bool);
  vtkGetMacro(DistributeBlocks, bool);
  vtkBooleanMacro(DistributeBlocks, bool);

  ///@{
  /**
   * This reader can cache the mesh points if they are time invariant.
   * They will be stored with a unique reference to their /base/zonename
   * and not be read in the file when doing unsteady analysis.
   */
  void SetCacheMesh(bool enable);
  vtkGetMacro(CacheMesh, bool);
  vtkBooleanMacro(CacheMesh, bool);

  ///@{
  /**
   * This reader can cache the mesh connectivities if they are time invariant.
   * They will be stored with a unique reference to their /base/zonename
   * and not be read in the file when doing unsteady analysis.
   */
  void SetCacheConnectivity(bool enable);
  vtkGetMacro(CacheConnectivity, bool);
  vtkBooleanMacro(CacheConnectivity, bool);

  ///@{
  /**
   * Set/get the communication object used to relay a list of files
   * from the rank 0 process to all others. This is the only interprocess
   * communication required by vtkPExodusIIReader.
   */
  void SetController(vtkMultiProcessController* c);
  vtkGetObjectMacro(Controller, vtkMultiProcessController);
  ///@}

  /**
   * Sends metadata (that read from the input file, not settings modified
   * through this API) from the rank 0 node to all other processes in a job.
   */
  void Broadcast(vtkMultiProcessController* ctrl);

  /**
   * Key used to put a family name in the meta-data associated with a node
   */
  static vtkInformationStringKey* FAMILY();

protected:
  vtkCGNSReader();
  ~vtkCGNSReader() override;

  int FillOutputPortInformation(int port, vtkInformation* info) override;
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  int GetCurvilinearZone(
    int base, int zone, int cell_dim, int phys_dim, void* zsize, vtkMultiBlockDataSet* mbase);

  int GetUnstructuredZone(
    int base, int zone, int cell_dim, int phys_dim, void* zsize, vtkMultiBlockDataSet* mbase);

  vtkMultiProcessController* Controller = nullptr;
  vtkIdType ProcRank = 0;
  vtkIdType ProcSize = 1;

  vtkNew<vtkDataArraySelection> BaseSelection;
  vtkNew<vtkDataArraySelection> FamilySelection;

  vtkNew<vtkDataArraySelection> CellDataArraySelection;
  vtkNew<vtkDataArraySelection> FaceDataArraySelection;
  vtkNew<vtkDataArraySelection> PointDataArraySelection;

private:
  vtkCGNSReader(const vtkCGNSReader&) = delete;
  void operator=(const vtkCGNSReader&) = delete;

  std::string FileName = "";
  int DataLocation = vtkCGNSReader::CELL_DATA;
  bool LoadBndPatch = false;         // option to set section loading for unstructured grid
  bool LoadMesh = true;              // option to enable/disable mesh loading
  int DoublePrecisionMesh = 1;       // option to set mesh loading to double precision
  int CreateEachSolutionAsBlock = 0; // debug option to create
  bool IgnoreFlowSolutionPointers = false;
  bool UseUnsteadyPattern = false;
  bool DistributeBlocks = true;
  bool CacheMesh = false;
  bool CacheConnectivity = false;
  bool Use3DVector = true;

  // For internal cgio calls (low level IO)
  int cgioNum;      // cgio file reference
  double rootId;    // id of root node
  double currentId; // id of node currently being read (zone)

  unsigned int NumberOfBases = 0;
  int ActualTimeStep = 0;

  class vtkPrivate;
  vtkPrivate* Internals;
  friend class vtkPrivate;
};

#endif // vtkCGNSReader_h
