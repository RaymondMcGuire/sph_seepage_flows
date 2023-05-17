
#ifndef VTKIOCHEMISTRY_EXPORT_H
#define VTKIOCHEMISTRY_EXPORT_H

#ifdef VTKIOCHEMISTRY_STATIC_DEFINE
#  define VTKIOCHEMISTRY_EXPORT
#  define VTKIOCHEMISTRY_NO_EXPORT
#else
#  ifndef VTKIOCHEMISTRY_EXPORT
#    ifdef IOChemistry_EXPORTS
        /* We are building this library */
#      define VTKIOCHEMISTRY_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define VTKIOCHEMISTRY_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef VTKIOCHEMISTRY_NO_EXPORT
#    define VTKIOCHEMISTRY_NO_EXPORT 
#  endif
#endif

#ifndef VTKIOCHEMISTRY_DEPRECATED
#  define VTKIOCHEMISTRY_DEPRECATED __declspec(deprecated)
#endif

#ifndef VTKIOCHEMISTRY_DEPRECATED_EXPORT
#  define VTKIOCHEMISTRY_DEPRECATED_EXPORT VTKIOCHEMISTRY_EXPORT VTKIOCHEMISTRY_DEPRECATED
#endif

#ifndef VTKIOCHEMISTRY_DEPRECATED_NO_EXPORT
#  define VTKIOCHEMISTRY_DEPRECATED_NO_EXPORT VTKIOCHEMISTRY_NO_EXPORT VTKIOCHEMISTRY_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef VTKIOCHEMISTRY_NO_DEPRECATED
#    define VTKIOCHEMISTRY_NO_DEPRECATED
#  endif
#endif

/* AutoInit implementations. */
#ifdef vtkIOChemistry_AUTOINIT_INCLUDE
#include vtkIOChemistry_AUTOINIT_INCLUDE
#endif
#ifdef vtkIOChemistry_AUTOINIT
#include "vtkAutoInit.h"
VTK_MODULE_AUTOINIT(vtkIOChemistry)
#endif

#endif /* VTKIOCHEMISTRY_EXPORT_H */
