
#ifndef VTKIOIOSS_EXPORT_H
#define VTKIOIOSS_EXPORT_H

#ifdef VTKIOIOSS_STATIC_DEFINE
#  define VTKIOIOSS_EXPORT
#  define VTKIOIOSS_NO_EXPORT
#else
#  ifndef VTKIOIOSS_EXPORT
#    ifdef IOIOSS_EXPORTS
        /* We are building this library */
#      define VTKIOIOSS_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define VTKIOIOSS_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef VTKIOIOSS_NO_EXPORT
#    define VTKIOIOSS_NO_EXPORT 
#  endif
#endif

#ifndef VTKIOIOSS_DEPRECATED
#  define VTKIOIOSS_DEPRECATED __declspec(deprecated)
#endif

#ifndef VTKIOIOSS_DEPRECATED_EXPORT
#  define VTKIOIOSS_DEPRECATED_EXPORT VTKIOIOSS_EXPORT VTKIOIOSS_DEPRECATED
#endif

#ifndef VTKIOIOSS_DEPRECATED_NO_EXPORT
#  define VTKIOIOSS_DEPRECATED_NO_EXPORT VTKIOIOSS_NO_EXPORT VTKIOIOSS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef VTKIOIOSS_NO_DEPRECATED
#    define VTKIOIOSS_NO_DEPRECATED
#  endif
#endif

#endif /* VTKIOIOSS_EXPORT_H */
