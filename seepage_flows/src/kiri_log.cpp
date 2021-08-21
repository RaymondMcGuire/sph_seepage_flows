/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-09-27 03:01:47
 * @LastEditTime: 2020-10-25 14:33:09
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_log.cpp
 */

#include <kiri_log.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace KIRI
{
    std::shared_ptr<spdlog::logger> KiriLog::mLogger;

    void KiriLog::Init()
    {
        spdlog::set_pattern("%^[%T] %n: %v%$");

        mLogger = spdlog::stdout_color_mt("KIRI_LOG");
        mLogger->set_level(spdlog::level::trace);
    }
} // namespace KIRI
