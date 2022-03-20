/***
 * @Author: Jayden Zhang
 * @Date: 2020-09-27 02:54:00
 * @LastEditTime: 2021-08-21 19:46:42
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \sph_seepage_flows\seepage_flows\include\kiri_utils.h
 */

#ifndef _KIRI_UTILS_H_
#define _KIRI_UTILS_H_

#pragma once

#include <root_directory.h>
#include <partio/Partio.h>

#include <kiri_pch.h>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>

namespace KIRI
{
    String UInt2Str4Digit(
        UInt input)
    {
        char output[5];
        snprintf(output, 5, "%04d", input);
        return String(output);
    };

    Vec_Float4 ReadBgeoFileForGPU(
        String Folder,
        String Name,
        bool flip_yz = false)
    {
        String root_folder = "bgeo";
        String extension = ".bgeo";
        String file_path = String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
        Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

        Partio::ParticleAttribute pos_attr;
        Partio::ParticleAttribute pscale_attr;
        if (!data->attributeInfo("position", pos_attr) || (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) || pos_attr.count != 3)
        {
            KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
        }

        bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

        Vec_Float4 pos_array;
        for (Int i = 0; i < data->numParticles(); i++)
        {
            const float *pos = data->data<float>(pos_attr, i);
            if (pscaleLoaded)
            {
                const float *pscale = data->data<float>(pscale_attr, i);
                if (i == 0)
                {
                    KIRI_LOG_INFO("pscale={0}", *pscale);
                }

                if (flip_yz)
                {
                    pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
                }
                else
                {
                    pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
                }
            }
            else
            {
                if (flip_yz)
                {
                    pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
                }
                else
                {
                    pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
                }
            }
        }

        data->release();

        return pos_array;
    }

    Vec_Float4 ReadMultiBgeoFilesForGPU(
        Vec_String folders,
        Vec_String file_names,
        bool flip_yz = false)
    {
        auto root_folder = "bgeo";
        auto extension = ".bgeo";

        Vec_Float4 pos_array;
        for (auto n = 0; n < folders.size(); n++)
        {
            auto file_path = String(DB_PBR_PATH) + root_folder + "/" + folders[n] + "/" + file_names[n] + extension;
            Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

            Partio::ParticleAttribute pos_attr;
            Partio::ParticleAttribute pscale_attr;

            if (!data->attributeInfo("position", pos_attr) || (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) || pos_attr.count != 3)
                KIRI_LOG_ERROR("File={0}, Failed to Get Proper Position Attribute", file_names[n]);

            auto pscale_loaded = data->attributeInfo("pscale", pscale_attr);

            for (auto i = 0; i < data->numParticles(); i++)
            {
                auto *pos = data->data<float>(pos_attr, i);
                if (pscale_loaded)
                {
                    auto *pscale = data->data<float>(pscale_attr, i);

                    if (i == 0)
                        KIRI_LOG_INFO("pscale={0}", *pscale);

                    if (flip_yz)
                        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
                    else
                        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
                }
                else
                {
                    if (flip_yz)
                        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));

                    else
                        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
                }
            }

            KIRI_LOG_INFO("Loaded Bgeo File={0}, Number of Particles={1}", file_names[n], data->numParticles());

            data->release();
        }

        return pos_array;
    }

    void ExportBgeoFileCUDA(
        String folder_path,
        String file_name,
        float3 *positions,
        float3 *colors,
        float *radius,
        size_t *labels,
        UInt particles_num)
    {
        String export_file = folder_path + "/" + file_name + ".bgeo";

        try
        {

            struct stat info;

            if (stat(folder_path.c_str(), &info) != 0)
            {
                std::error_code ec;
                bool success = std::filesystem::create_directories(folder_path, ec);
                if (!success)
                {
                    std::cout << ec.message() << std::endl;
                }
            }

            Partio::ParticlesDataMutable *p = Partio::create();
            Partio::ParticleAttribute position_attr = p->addAttribute("position", Partio::VECTOR, 3);
            Partio::ParticleAttribute color_attr = p->addAttribute("Cd", Partio::FLOAT, 3);
            Partio::ParticleAttribute pscale_attr = p->addAttribute("pscale", Partio::FLOAT, 1);
            Partio::ParticleAttribute label_attr = p->addAttribute("label", Partio::INT, 1);

            // transfer GPU data to CPU
            size_t fbytes = particles_num * sizeof(float);
            size_t f3bytes = particles_num * sizeof(float3);
            size_t uintbytes = particles_num * sizeof(size_t);

            float3 *cpu_positions = (float3 *)malloc(f3bytes);
            float3 *cpu_colors = (float3 *)malloc(f3bytes);
            float *cpu_radius = (float *)malloc(fbytes);
            size_t *cpu_labels = (size_t *)malloc(uintbytes);

            cudaMemcpy(cpu_positions, positions, f3bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_colors, colors, f3bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_radius, radius, fbytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_labels, labels, uintbytes, cudaMemcpyDeviceToHost);

            for (UInt i = 0; i < particles_num; i++)
            {
                Int particle = p->addParticle();
                float *pos = p->dataWrite<float>(position_attr, particle);
                float *col = p->dataWrite<float>(color_attr, particle);
                float *pscale = p->dataWrite<float>(pscale_attr, particle);
                int *label = p->dataWrite<int>(label_attr, particle);

                pos[0] = cpu_positions[i].x;
                pos[1] = cpu_positions[i].y;
                pos[2] = cpu_positions[i].z;
                col[0] = cpu_colors[i].x;
                col[1] = cpu_colors[i].y;
                col[2] = cpu_colors[i].z;

                // TODO
                *pscale = cpu_radius[i];

                *label = cpu_labels[i];
            }

            Partio::write(export_file.c_str(), *p);

            p->release();

            free(cpu_positions);
            free(cpu_colors);
            free(cpu_labels);
            free(cpu_radius);

            KIRI_LOG_INFO("Successfully saved bgeo file:{0}", export_file);
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }
    }

} // namespace KIRI
#endif