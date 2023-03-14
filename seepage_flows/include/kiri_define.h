/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:41:33
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_define.h
 */

#ifndef _KIRI_DEFINE_H_
#define _KIRI_DEFINE_H_
#pragma once

#if defined(_WIN32) || defined(_WIN64)
#define KIRI_WINDOWS
#elif defined(__APPLE__)
#define KIRI_APPLE
#endif

#ifdef KIRI_WINDOWS
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#ifdef KIRI_APPLE
#include <stddef.h>
#include <sys/types.h>
#endif

#if defined(DEBUG) || defined(_DEBUG)
#define KIRI_DEBUG_MODE
#define RELEASE false
#define PUBLISH false
#else
#define KIRI_RELEASE_MODE
#define RELEASE true
#define PUBLISH false
#endif

#ifdef KIRI_DEBUG_MODE
#define KIRI_ENABLE_ASSERTS
#endif

// ASSERTION
#ifdef KIRI_ENABLE_ASSERTS
#ifdef KIRI_WINDOWS
#include <cassert>
#define KIRI_ASSERT(x) assert(x)
#endif
#else
#define KIRI_ASSERT(x)
#endif

#define APP_NAME "KIRI"
//#define DOUBLE_PRECISION

#endif // _KIRI_DEFINE_H_