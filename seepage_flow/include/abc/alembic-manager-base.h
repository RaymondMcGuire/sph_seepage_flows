/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-17 00:36:03
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-19 15:54:16
 * @FilePath: \Kiri\KiriExamples\include\abc\alembic-manager-base.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef PBF_ALEMBIC_MANAGER_BASE_HPP
#define PBF_ALEMBIC_MANAGER_BASE_HPP

#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <cstdint>
#include <string>

class AlembicManagerBase {
public:
  AlembicManagerBase(const std::string &output_file_path,
                     const double delta_time, const std::string &object_name)
      : m_archive(Alembic::AbcCoreOgawa::WriteArchive(),
                  output_file_path.c_str()) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    const TimeSampling time_sampling(delta_time, 0);
    const std::uint32_t time_sampling_index =
        m_archive.addTimeSampling(time_sampling);

    m_points = OPoints(OObject(m_archive, kTop), object_name.c_str());
    m_points.getSchema().setTimeSampling(time_sampling_index);
  }

protected:
  bool m_is_first = true;

  Alembic::Abc::OArchive m_archive;
  Alembic::AbcGeom::OPoints m_points;
};

#endif
