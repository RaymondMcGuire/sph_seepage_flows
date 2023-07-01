/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkInteractorEventRecorder.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   vtkInteractorEventRecorder
 * @brief   record and play VTK events passing through a vtkRenderWindowInteractor
 *
 *
 * vtkInteractorEventRecorder records all VTK events invoked from a
 * vtkRenderWindowInteractor. The events are recorded to a
 * file. vtkInteractorEventRecorder can also be used to play those events
 * back and invoke them on an vtkRenderWindowInteractor. (Note: the events
 * can also be played back from a file or string.)
 *
 * Event client data can be recorded as args and will be provided on replay.
 * The following event current support data to be recorded:
 *  - DropFilesEvents: record a string array
 *
 * The format of the event file is simple. It is:
 *  EventName X Y ctrl shift keycode repeatCount keySym dataType [dataNum] [dataVal] [dataVal]
 * The format also allows "#" comments.
 * dataType is defined as follows:
 *  - 0 -> None
 *  - 1 -> StringArray
 *
 * @sa
 * vtkInteractorObserver vtkCallback
 */

#ifndef vtkInteractorEventRecorder_h
#define vtkInteractorEventRecorder_h

#include "vtkDeprecation.h" // For VTK_DEPRECATED_IN_9_2_0
#include "vtkInteractorObserver.h"
#include "vtkRenderingCoreModule.h" // For export macro

class vtkStringArray;

// The superclass that all commands should be subclasses of
class VTKRENDERINGCORE_EXPORT vtkInteractorEventRecorder : public vtkInteractorObserver
{
public:
  static vtkInteractorEventRecorder* New();
  vtkTypeMacro(vtkInteractorEventRecorder, vtkInteractorObserver);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // enumeration of data type
  enum class vtkEventDataType : int
  {
    None = 0,
    StringArray
  };

  // Satisfy the superclass API. Enable/disable listening for events.
  void SetEnabled(int) override;
  void SetInteractor(vtkRenderWindowInteractor* iren) override;

  ///@{
  /**
   * Set/Get the name of a file events should be written to/from.
   * Will be ignored once record/play has been called.
   */
  vtkSetFilePathMacro(FileName);
  vtkGetFilePathMacro(FileName);
  ///@}

  /**
   * Invoke this method to begin recording events. The events will be
   * recorded to the filename indicated.
   * Once record has been called once, filename will be ignored.
   */
  void Record();

  /**
   * Invoke this method to begin playing events from the current position.
   * The events will be played back from the filename indicated.
   * Once play has been called once, filename will be ignored.
   */
  void Play();

  /**
   * Invoke this method to stop recording/playing events.
   */
  void Stop();

  /**
   * Invoke this method to clear recording/playing stream and be able to open
   * another file using the same recorder.
   */
  void Clear();

  /**
   * Rewind the play stream to the beginning of the file.
   */
  void Rewind();

  ///@{
  /**
   * Enable reading from an InputString as compared to the default
   * behavior, which is to read from a file.
   */
  vtkSetMacro(ReadFromInputString, vtkTypeBool);
  vtkGetMacro(ReadFromInputString, vtkTypeBool);
  vtkBooleanMacro(ReadFromInputString, vtkTypeBool);
  ///@}

  ///@{
  /**
   * Set/Get the string to read from.
   */
  vtkSetStringMacro(InputString);
  vtkGetStringMacro(InputString);
  ///@}

protected:
  vtkInteractorEventRecorder();
  ~vtkInteractorEventRecorder() override;

  // file to read/write from
  char* FileName;

  // listens to delete events
  vtkCallbackCommand* DeleteEventCallbackCommand;

  // control whether to read from string
  vtkTypeBool ReadFromInputString;
  char* InputString;

  // for reading and writing
  istream* InputStream;
  ostream* OutputStream;

  // methods for processing events
  static void ProcessCharEvent(
    vtkObject* object, unsigned long event, void* clientdata, void* calldata);
  static void ProcessDeleteEvent(
    vtkObject* object, unsigned long event, void* clientdata, void* calldata);
  static void ProcessEvents(
    vtkObject* object, unsigned long event, void* clientdata, void* calldata);

  virtual void WriteEvent(const char* event, int pos[2], int modifiers, int keyCode,
    int repeatCount, char* keySym, void* callData = nullptr);

  VTK_DEPRECATED_IN_9_2_0(
    "This method was not used at all and has been replaced by ReadEvent(const std::string&)")
  virtual void ReadEvent(){};

  /**
   * A method that parse a event line and invoke the corresponding event
   */
  virtual void ReadEvent(const std::string& line);

  // Manage the state of the recorder
  int State;
  enum WidgetState
  {
    Start = 0,
    Playing,
    Recording
  };

  // Associate a modifier with a bit
  enum ModifierKey
  {
    ShiftKey = 1,
    ControlKey = 2,
    AltKey = 4
  };

  static float StreamVersion;
  float CurrentStreamVersion;

private:
  vtkInteractorEventRecorder(const vtkInteractorEventRecorder&) = delete;
  void operator=(const vtkInteractorEventRecorder&) = delete;
};

#endif /* vtkInteractorEventRecorder_h */
