/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-12-13 20:57:45
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-14 19:24:42
 * @FilePath: \sph_seepage_flows\seepage_flows\include\kiri_utils.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
#include <regex>

#include <file_io.hpp>

namespace KIRI
{

    struct Vec3f
    {
        float data[3];
    };

    struct Vec2i
    {
        int data[2];
    };

    String UInt2Str4Digit(
        UInt input)
    {
        char output[5];
        snprintf(output, 5, "%04d", input);
        return String(output);
    };

    template <typename T>
    inline void swapByteOrder(T *v)
    {
        constexpr size_t n = sizeof(T);
        uint8_t out[n];
        for (unsigned int c = 0; c < n; c++)
            out[c] = reinterpret_cast<uint8_t *>(v)[n - c - 1];
        std::memcpy(v, out, n);
    }

    void Partio2VTK(
        const std::string &file_path,
        const Partio::ParticlesDataMutable *partioData)
    {
        const unsigned int numParticles = partioData->numParticles();
        if (0 == numParticles)
            return;

        KIRI_LOG_INFO("start convert bgeo to vtk !");

        std::ofstream outfile{file_path, std::ios::binary};

        outfile << "# vtk DataFile Version 4.1\n";
        outfile << "Seepage Flow Framework Particles Data\n";
        outfile << "BINARY\n";
        outfile << "DATASET UNSTRUCTURED_GRID\n";

        // find indices of position and ID attribute
        unsigned int posIndex = 0xffffffff;
        unsigned int idIndex = 0xffffffff;
        for (int i = 0; i < partioData->numAttributes(); i++)
        {
            Partio::ParticleAttribute attr;
            partioData->attributeInfo(i, attr);
            if (attr.name == "position")
                posIndex = i;
            else if (attr.name == "id")
                idIndex = i;
        }

        // export position attribute as POINTS
        if (0xffffffff != posIndex)
        {
            // copy from partio data
            std::vector<Vec3f> positions;
            positions.reserve(numParticles);
            Partio::ParticleAttribute attr;
            partioData->attributeInfo(posIndex, attr);
            for (unsigned int i = 0u; i < numParticles; i++)
            {
                auto partio_pos = partioData->data<float>(attr, i);
                Vec3f v3;
                v3.data[0] = partio_pos[0];
                v3.data[1] = partio_pos[1];
                v3.data[2] = partio_pos[2];
                positions.emplace_back(v3);
            }

            // swap endianess
            for (unsigned int i = 0; i < numParticles; i++)
                for (unsigned int c = 0; c < 3; c++)
                    swapByteOrder(&positions[i].data[c]);

            // export to vtk
            outfile << "POINTS " << numParticles << " float\n";
            outfile.write(reinterpret_cast<char *>(positions[0].data), 3 * numParticles * sizeof(float));
            outfile << "\n";
        }
        else
        {
            KIRI_LOG_ERROR("not found particle data!");
            return;
        }

        // export particle IDs as CELLS
        {
            std::vector<Vec2i> cells;
            cells.reserve(numParticles);
            int nodes_per_cell_swapped = 1;
            swapByteOrder(&nodes_per_cell_swapped);
            if (0xffffffff != idIndex)
            {
                // load IDs from partio
                Partio::ParticleAttribute attr;
                partioData->attributeInfo(idIndex, attr);
                for (unsigned int i = 0u; i < numParticles; i++)
                {
                    int idSwapped = *partioData->data<int>(attr, i);
                    swapByteOrder(&idSwapped);
                    Vec2i v2i;
                    v2i.data[0] = nodes_per_cell_swapped;
                    v2i.data[1] = idSwapped;
                    cells.emplace_back(v2i);
                }
            }
            else
            {
                // generate IDs
                for (unsigned int i = 0u; i < numParticles; i++)
                {
                    int idSwapped = i;
                    swapByteOrder(&idSwapped);
                    Vec2i v2i;
                    v2i.data[0] = nodes_per_cell_swapped;
                    v2i.data[1] = idSwapped;
                    cells.emplace_back(v2i);
                }
            }

            // particles are cells with one element and the index of the particle
            outfile << "CELLS " << numParticles << " " << 2 * numParticles << "\n";
            outfile.write(reinterpret_cast<char *>(cells[0].data), 2 * numParticles * sizeof(int));
            outfile << "\n";
        }

        // export cell types
        {
            // the type of a particle cell is always 1
            std::vector<int> cellTypes;
            unsigned int cellTypeSwapped = 1;
            swapByteOrder(&cellTypeSwapped);
            cellTypes.resize(numParticles, cellTypeSwapped);
            outfile << "CELL_TYPES " << numParticles << "\n";
            outfile.write(reinterpret_cast<char *>(cellTypes.data()), numParticles * sizeof(int));
            outfile << "\n";
        }

        // write additional attributes as per-particle data
        outfile << "POINT_DATA " << numParticles << "\n";
        // per point fields (all attributes except for positions and IDs)
        const unsigned int numFields = partioData->numAttributes() - static_cast<int>(0xffffffff != posIndex) - static_cast<int>(0xffffffff != idIndex);
        outfile << "FIELD FieldData " << std::to_string(numFields) << "\n";
        // iterate over attributes
        for (int a = 0; a < partioData->numAttributes(); a++)
        {
            if (posIndex == a || idIndex == a)
                continue;

            Partio::ParticleAttribute attr;
            partioData->attributeInfo(a, attr);
            std::string attrNameVTK;
            std::regex_replace(std::back_inserter(attrNameVTK), attr.name.begin(), attr.name.end(), std::regex("\\s+"), "_");
            // write header information
            outfile << attrNameVTK << " " << attr.count << " " << numParticles;
            // write depending on data type
            if (attr.type == Partio::ParticleAttributeType::FLOAT)
            {
                outfile << " float\n";
                // copy from partio data
                std::vector<float> attrData;
                attrData.reserve(partioData->numParticles());
                for (unsigned int i = 0u; i < numParticles; i++)
                    attrData.emplace_back(*partioData->data<float>(attr, i));
                // swap endianess
                for (unsigned int i = 0; i < numParticles; i++)
                    swapByteOrder(&attrData[i]);
                // export to vtk
                outfile.write(reinterpret_cast<char *>(attrData.data()), numParticles * sizeof(float));
            }
            else if (attr.type == Partio::ParticleAttributeType::VECTOR)
            {
                outfile << " float\n";
                // copy from partio data
                std::vector<Vec3f> attrData;
                attrData.reserve(partioData->numParticles());
                for (unsigned int i = 0u; i < numParticles; i++)
                {
                    Vec3f v3;
                    auto partio_attr = partioData->data<float>(attr, i);
                    v3.data[0] = partio_attr[0];
                    v3.data[1] = partio_attr[1];
                    v3.data[2] = partio_attr[2];
                    attrData.emplace_back(v3);
                }

                // swap endianess
                for (unsigned int i = 0; i < numParticles; i++)
                    for (unsigned int c = 0; c < 3; c++)
                        swapByteOrder(&attrData[i].data[c]);

                // export to vtk
                outfile.write(reinterpret_cast<char *>(attrData[0].data), 3 * numParticles * sizeof(float));
            }
            else if (attr.type == Partio::ParticleAttributeType::INT)
            {
                outfile << " int\n";
                // copy from partio data
                std::vector<int> attrData;
                attrData.reserve(partioData->numParticles());
                for (unsigned int i = 0u; i < numParticles; i++)
                    attrData.emplace_back(*partioData->data<int>(attr, i));
                // swap endianess
                for (unsigned int i = 0; i < numParticles; i++)
                    swapByteOrder(&attrData[i]);
                // export to vtk
                outfile.write(reinterpret_cast<char *>(attrData.data()), numParticles * sizeof(int));
            }
            else
            {
                KIRI_LOG_INFO("unsupport type!");
                continue;
            }
            // end of block
            outfile << "\n";
        }
        outfile.close();
    }

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
            auto file_path = std::string(DB_PBR_PATH) + root_folder + "/" + folders[n] + "/" + file_names[n] + extension;
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
        const std::string &folder_path,
        const std::string &file_name,
        const float3 *positions,
        const float3 *velocities,
        const float3 *colors,
        const float *radius,
        const float *pressure,
        const size_t *labels,
        const size_t particles_num)
    {
        std::string export_bgeo = folder_path + "/" + file_name + ".bgeo";
        // std::string export_vtk = folder_path + "/" + file_name + ".vtk";
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
            Partio::ParticleAttribute velocity_attr = p->addAttribute("v", Partio::VECTOR, 3);
            Partio::ParticleAttribute color_attr = p->addAttribute("Cd", Partio::FLOAT, 3);
            Partio::ParticleAttribute pscale_attr = p->addAttribute("pscale", Partio::FLOAT, 1);
            Partio::ParticleAttribute pressure_attr = p->addAttribute("pressure", Partio::FLOAT, 1);
            Partio::ParticleAttribute label_attr = p->addAttribute("label", Partio::INT, 1);

            // transfer GPU data to CPU
            size_t fbytes = particles_num * sizeof(float);
            size_t f3bytes = particles_num * sizeof(float3);
            size_t uintbytes = particles_num * sizeof(size_t);

            float3 *cpu_positions = (float3 *)malloc(f3bytes);
            float3 *cpu_velocities = (float3 *)malloc(f3bytes);
            float3 *cpu_colors = (float3 *)malloc(f3bytes);
            float *cpu_radius = (float *)malloc(fbytes);
            float *cpu_pressure = (float *)malloc(fbytes);
            size_t *cpu_labels = (size_t *)malloc(uintbytes);

            cudaMemcpy(cpu_positions, positions, f3bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_velocities, velocities, f3bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_colors, colors, f3bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_radius, radius, fbytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_pressure, pressure, fbytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_labels, labels, uintbytes, cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < particles_num; i++)
            {
                Int particle = p->addParticle();
                float *pos = p->dataWrite<float>(position_attr, particle);
                float *vel = p->dataWrite<float>(velocity_attr, particle);
                float *col = p->dataWrite<float>(color_attr, particle);
                float *pscale = p->dataWrite<float>(pscale_attr, particle);
                float *press = p->dataWrite<float>(pressure_attr, particle);
                int *label = p->dataWrite<int>(label_attr, particle);

                pos[0] = cpu_positions[i].x;
                pos[1] = cpu_positions[i].y;
                pos[2] = cpu_positions[i].z;

                vel[0] = cpu_velocities[i].x;
                vel[1] = cpu_velocities[i].y;
                vel[2] = cpu_velocities[i].z;

                col[0] = cpu_colors[i].x;
                col[1] = cpu_colors[i].y;
                col[2] = cpu_colors[i].z;

                *pscale = cpu_radius[i];

                *press = cpu_pressure[i];

                *label = cpu_labels[i];
            }

            // Partio2VTK(export_vtk, p);

            Partio::write(export_bgeo.c_str(), *p);

            p->release();

            free(cpu_positions);
            free(cpu_colors);
            free(cpu_labels);
            free(cpu_radius);

            KIRI_LOG_INFO("successfully saved bgeo file:{0}", export_bgeo);
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }
    }

    void ExportData2DatFile(
        const std::string &folder_path,
        const std::string &file_name,
        const float3 *positions,
        const float3 *velocities,
        const size_t *labels,
        const float *density,
        const float3 *acc,
        const size_t *id,
        const float *saturation,
        const float young,
        const float cd,
        const size_t particles_num)
    {
        std::string export_dat = folder_path + "/" + file_name + ".dat";
        auto dat_file = evt::FileIO::plainText(export_dat);
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

            // transfer GPU data to CPU
            size_t fbytes = particles_num * sizeof(float);
            size_t f3bytes = particles_num * sizeof(float3);
            size_t uintbytes = particles_num * sizeof(size_t);

            float3 *cpu_positions = (float3 *)malloc(f3bytes);
            float3 *cpu_velocities = (float3 *)malloc(f3bytes);
            float3 *cpu_acc = (float3 *)malloc(f3bytes);

            float *cpu_density = (float *)malloc(fbytes);
            float *cpu_saturation = (float *)malloc(fbytes);

            size_t *cpu_id = (size_t *)malloc(uintbytes);
            size_t *cpu_labels = (size_t *)malloc(uintbytes);

            cudaMemcpy(cpu_positions, positions, f3bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_velocities, velocities, f3bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_acc, acc, f3bytes, cudaMemcpyDeviceToHost);

            cudaMemcpy(cpu_density, density, fbytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_saturation, saturation, fbytes, cudaMemcpyDeviceToHost);

            cudaMemcpy(cpu_id, id, uintbytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_labels, labels, uintbytes, cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < particles_num; i++)
            {
                if (i != 0)
                    dat_file.write("\n");

                dat_file.write(cpu_positions[i].x);
                dat_file.write(" ");
                dat_file.write(cpu_positions[i].y);
                dat_file.write(" ");
                dat_file.write(cpu_positions[i].z);
                dat_file.write(" ");

                dat_file.write(cpu_velocities[i].x);
                dat_file.write(" ");
                dat_file.write(cpu_velocities[i].y);
                dat_file.write(" ");
                dat_file.write(cpu_velocities[i].z);
                dat_file.write(" ");

                dat_file.write(cpu_labels[i]);
                dat_file.write(" ");

                dat_file.write(cpu_density[i]);
                dat_file.write(" ");

                dat_file.write(cpu_acc[i].x);
                dat_file.write(" ");
                dat_file.write(cpu_acc[i].y);
                dat_file.write(" ");
                dat_file.write(cpu_acc[i].z);
                dat_file.write(" ");

                dat_file.write(young);
                dat_file.write(" ");

                dat_file.write(cd);
                dat_file.write(" ");

                dat_file.write(cpu_id[i]);
                dat_file.write(" ");

                dat_file.write(cpu_saturation[i]);
            }

            free(cpu_positions);
            free(cpu_acc);
            free(cpu_labels);
            free(cpu_velocities);
            free(cpu_id);
            free(cpu_density);

            KIRI_LOG_INFO("successfully saved dat file:{0}", export_dat);
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }
    }

} // namespace KIRI
#endif