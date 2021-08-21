/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-09-27 02:54:00
 * @LastEditTime: 2021-02-20 18:50:37
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_log.h
 */

#ifndef _KIRI_LOG_H_
#define _KIRI_LOG_H_

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace KIRI
{
    class KiriLog
    {
    public:
        static void Init();

        inline static std::shared_ptr<spdlog::logger> &GetLogger() { return mLogger; };

    private:
        static std::shared_ptr<spdlog::logger> mLogger;
    };
} // namespace KIRI

#define KIRI_LOG_TRACE(...) ::KIRI::KiriLog::GetLogger()->trace(__VA_ARGS__)
#define KIRI_LOG_INFO(...) ::KIRI::KiriLog::GetLogger()->info(__VA_ARGS__)
#define KIRI_LOG_DEBUG(...) ::KIRI::KiriLog::GetLogger()->debug(__VA_ARGS__)
#define KIRI_LOG_WARN(...) ::KIRI::KiriLog::GetLogger()->warn(__VA_ARGS__)
#define KIRI_LOG_ERROR(...) ::KIRI::KiriLog::GetLogger()->error(__VA_ARGS__)
#define KIRI_LOG_FATAL(...) ::KIRI::KiriLog::GetLogger()->fatal(__VA_ARGS__)

#endif