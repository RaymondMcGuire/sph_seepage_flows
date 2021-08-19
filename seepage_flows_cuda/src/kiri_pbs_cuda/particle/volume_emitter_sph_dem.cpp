/*** 
 * @Author: Xu.WANG
 * @Date: 2020-09-14 10:03:28
 * @LastEditTime: 2021-01-17 16:17:30
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\volume_emitter_sph_dem.cpp
 */

#include <kiri_pbs_cuda/particle/volume_emitter_sph_dem.h>
#include <random>
#include <array>

SphDemDataPtr VolumeEmitterSphDem::EmitSphBoxVolume(SphBoxVolumeParams Params)
{
    SphDemDataPtr data_block = std::make_shared<SphDemData>();

    uint counter = SPH_DEM_DEMO_PARAMS.NumOfParticles;
    int3 nn = Params.BoxSize;
    float3 lower = Params.Lower;
    float radius = SPH_DEM_DEMO_PARAMS.SphParticleRadius;
    float offset = 2.f * radius;
    float3 start_point = lower + make_float3(radius);

    for (size_t i = 0; i < nn.x; i++)
    {
        for (size_t j = 0; j < nn.y; j++)
        {
            for (size_t k = 0; k < nn.z; k++)
            {
                float4 p = make_float4(start_point.x + i * offset, start_point.y + j * offset, start_point.z + k * offset, SPH_DEM_DEMO_PARAMS.SphParticleRadius);

                data_block->id().emplace_back(counter++);
                data_block->label().emplace_back(0);
                data_block->positions().emplace_back(p);
                data_block->fpositions().emplace_back(p);
                data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.SphInitBoxColor, 0.f));
                data_block->masses().emplace_back(SPH_DEM_DEMO_PARAMS.SphRestMass);
                data_block->inertia().emplace_back(0.f);
            }
        }
    }

    return data_block;
}

SphDemDataPtr VolumeEmitterSphDem::EmitPlaneVolume(SphDynamicEmitterParams Params)
{
    SphDemDataPtr data_block = std::make_shared<SphDemData>();

    float3 emitter_vel = Params.EmitVelocity;
    float3 axis = normalize(emitter_vel);
    float3 e1 = make_float3(1.f);
    if (abs(axis.x) == 1.f && abs(axis.y) == 0.f && abs(axis.z) == 0.f)
    {
        e1 = normalize(cross(axis, make_float3(0.f, 1.f, 0.f)));
    }
    else
    {
        e1 = normalize(cross(axis, make_float3(1.f, 0.f, 0.f)));
    }

    float3 e2 = normalize(cross(axis, e1));

    //printf("e1=(%.3f,%.3f,%.3f),e2=(%.3f,%.3f,%.3f)\n", expand(e1), expand(e2));

    float offset = SPH_DEM_DEMO_PARAMS.SphParticleRadius * 2.f;
    std::vector<float2> samples;

    if (Params.SquareShapedEmitter && Params.CustomDefine)
    {
        for (float i = -Params.EmitWidth; i < Params.EmitWidth; i += offset)
        {
            for (float j = -Params.EmitHeight; j < Params.EmitHeight; j += offset)
            {
                samples.emplace_back(make_float2(i, j));
            }
        }
    }
    else
    {
        for (float i = -Params.EmitRadius; i < Params.EmitRadius; i += offset)
        {
            for (float j = -Params.EmitRadius; j < Params.EmitRadius; j += offset)
            {
                samples.emplace_back(make_float2(i, j));
            }
        }
    }

    // check the max emit number
    uint emit_num = samples.size();
    if ((emit_num + SPH_DEM_DEMO_PARAMS.NumOfParticles) >= SPH_DEM_DEMO_PARAMS.MaxNumOfParticles)
    {
        emit_num = SPH_DEM_DEMO_PARAMS.MaxNumOfParticles - SPH_DEM_DEMO_PARAMS.NumOfParticles;

        //TODO optimize
        SPH_DEM_DEMO_PARAMS.EmitParticles = false;
    }

    uint counter = 0;
    for (uint i = 0; i < emit_num; i++)
    {
        float3 tp = Params.EmitPosition + samples[i].x * e1 + samples[i].y * e2;

        if (!Params.SquareShapedEmitter && length(tp - Params.EmitPosition) > Params.EmitRadius)
            continue;

        float4 p = make_float4(tp.x, tp.y, tp.z, SPH_DEM_DEMO_PARAMS.SphParticleRadius);

        data_block->id().emplace_back(counter++);
        data_block->label().emplace_back(0);
        data_block->positions().emplace_back(p);

        if (Params.SPHVisable)
        {
            data_block->fpositions().emplace_back(p);
        }
        else
        {
            data_block->fpositions().emplace_back(make_float4(p.x, p.y, p.z, 0.f));
        }

        data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.SphInitBoxColor, 0.f));
        data_block->masses().emplace_back(SPH_DEM_DEMO_PARAMS.SphRestMass);
        data_block->inertia().emplace_back(0.f);
    }

    return data_block;
}

SphDemDataPtr VolumeEmitterSphDem::EmitSingleBox(std::vector<float3> BoundaryData)
{
    SphDemDataPtr data_block = std::make_shared<SphDemData>();

    uint counter = 0;
    SPH_DEM_DEMO_PARAMS.NumOfPhases = 2;
    mSampler = std::make_shared<ParticlesSamplerBasic>();

    // sph particles
    int3 nn = SPH_DEM_DEMO_PARAMS.SphInitBoxSize;
    float3 lower = SPH_DEM_DEMO_PARAMS.SphInitBoxLowestPoint;
    float radius = SPH_DEM_DEMO_PARAMS.SphParticleRadius;
    float offset = 2.f * radius;
    float3 start_point = lower + make_float3(radius);

    for (size_t i = 0; i < nn.x; i++)
    {
        for (size_t j = 0; j < nn.y; j++)
        {
            for (size_t k = 0; k < nn.z; k++)
            {
                float4 p = make_float4(start_point.x + i * offset, start_point.y + j * offset, start_point.z + k * offset, SPH_DEM_DEMO_PARAMS.SphParticleRadius);

                data_block->id().emplace_back(counter++);
                data_block->label().emplace_back(0);
                data_block->positions().emplace_back(p);
                data_block->fpositions().emplace_back(p);
                data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.SphInitBoxColor, 0.f));
                data_block->masses().emplace_back(SPH_DEM_DEMO_PARAMS.SphRestMass);
                data_block->inertia().emplace_back(0.f);
            }
        }
    }

    // re-calc sph rest density
    KernelPoly6 poly6_kernel = KernelPoly6(SPH_DEM_DEMO_PARAMS.SphKernelRadius);
    float l = 2 * SPH_DEM_DEMO_PARAMS.SphParticleRadius;
    int n = (int)std::ceil(SPH_DEM_DEMO_PARAMS.SphKernelRadius / l) + 1;

    float r0 = 0.f;
    for (int x = -n; x <= n; ++x)
    {
        for (int y = -n; y <= n; ++y)
        {
            for (int z = -n; z <= n; ++z)
            {
                float3 rij = make_float3(x * l, y * l, z * l);
                r0 += SPH_DEM_DEMO_PARAMS.SphRestMass * poly6_kernel(length(rij));
            }
        }
    }
    SPH_DEM_DEMO_PARAMS.SphRestDensity = r0;

    // dem particles
    if (SPH_DEM_DEMO_PARAMS.DemInitEnable)
    {
        // dem particles
        // std::vector<DEMSphere> demParticles = mSampler->GetCloudSampling(SPH_DEM_DEMO_PARAMS.DemInitBoxLowestPoint, SPH_DEM_DEMO_PARAMS.DemInitBoxHighestPoint, SPH_DEM_DEMO_PARAMS.DemParticleRadius, 0.5f);

        // for (size_t i = 0; i < demParticles.size(); i++)
        // {
        //     data_block->id().emplace_back(counter++);
        //     data_block->label().emplace_back(1);

        //     data_block->positions().emplace_back(make_float4(demParticles[i].center, demParticles[i].radius));
        //     data_block->fpositions().emplace_back(make_float4(demParticles[i].center, demParticles[i].radius));
        //     data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.DemInitBoxColor, 0.5f));

        //     float mass = SPH_DEM_DEMO_PARAMS.DemRestDensity * ((4.f / 3.f) * M_PI * std::powf(demParticles[i].radius, 3.f));
        //     float inertia = 2.f / 5.f * mass * demParticles[i].radius * demParticles[i].radius;
        //     data_block->masses().emplace_back(mass);
        //     data_block->inertia().emplace_back(inertia);
        // }
        int3 nns = SPH_DEM_DEMO_PARAMS.DemInitBoxSize;
        float3 dem_lower = SPH_DEM_DEMO_PARAMS.DemInitBoxLowestPoint;
        float dem_radius = SPH_DEM_DEMO_PARAMS.DemParticleRadius;
        float dem_offset = 2.f * dem_radius;
        float3 dem_start_point = dem_lower + make_float3(dem_radius);

        std::random_device engine;
        std::mt19937 gen(engine());
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        float jitter = 0.1f * dem_radius;

        for (size_t i = 0; i < nns.x; i++)
        {
            for (size_t j = 0; j < nns.y; j++)
            {
                for (size_t k = 0; k < nns.z; k++)
                {
                    float3 dem_position = make_float3(dem_start_point.x + i * dem_offset, dem_start_point.y + j * dem_offset, dem_start_point.z + k * dem_offset) + jitter * normalize(make_float3(dist(engine), dist(engine), dist(engine)));

                    float4 p = make_float4(dem_position, dem_radius);

                    data_block->id().emplace_back(counter++);
                    data_block->label().emplace_back(1);
                    data_block->positions().emplace_back(p);
                    data_block->fpositions().emplace_back(p);
                    data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.DemInitBoxColor, 0.5f));
                    float mass = SPH_DEM_DEMO_PARAMS.DemRestDensity * ((4.f / 3.f) * M_PI * std::powf(dem_radius, 3.f));
                    float inertia = 2.f / 5.f * mass * dem_radius * dem_radius;
                    data_block->masses().emplace_back(mass);
                    data_block->inertia().emplace_back(inertia);
                }
            }
        }
    }

    SPH_DEM_DEMO_PARAMS.NumOfParticles = counter;

    // boundary
    //auto boundaries = mSampler->GetBoxSampling(SPH_DEM_DEMO_PARAMS.LowestPoint, SPH_DEM_DEMO_PARAMS.HighestPoint, SPH_DEM_DEMO_PARAMS.SphParticleRadius * 2.f);
    auto boundaries = mSampler->GetBoxSampling(SPH_DEM_DEMO_PARAMS.LowestPoint, SPH_DEM_DEMO_PARAMS.HighestPoint, SPH_DEM_DEMO_PARAMS.BoundaryParticleRadius * 2.f);

    for (size_t i = 0; i < boundaries.size(); i++)
    {
        data_block->bpositions().emplace_back(boundaries[i]);
    }

    for (size_t i = 0; i < BoundaryData.size(); i++)
    {
        data_block->bpositions().emplace_back(BoundaryData[i]);
    }

    SPH_DEM_DEMO_PARAMS.NumOfBoundaries = boundaries.size() + BoundaryData.size();

    return data_block;
}

SphDemDataPtr VolumeEmitterSphDem::EmitSingleBoxByMSM(std::vector<float3> BoundaryData)
{
    SphDemDataPtr data_block = std::make_shared<SphDemData>();

    uint counter = 0;
    SPH_DEM_DEMO_PARAMS.NumOfPhases = 2;
    mSampler = std::make_shared<ParticlesSamplerBasic>();

    // dem particles
    MSMPackPtr sp1 = std::make_shared<MSMPack>(MSM_S1, SPH_DEM_DEMO_PARAMS.DemParticleRadius);
    MSMPackPtr sp2 = std::make_shared<MSMPack>(MSM_L2, SPH_DEM_DEMO_PARAMS.DemParticleRadius);
    MSMPackPtr sp3 = std::make_shared<MSMPack>(MSM_L3, SPH_DEM_DEMO_PARAMS.DemParticleRadius);
    MSMPackPtr sp4 = std::make_shared<MSMPack>(MSM_C8, SPH_DEM_DEMO_PARAMS.DemParticleRadius);
    MSMPackPtr sp5 = std::make_shared<MSMPack>(MSM_M7, SPH_DEM_DEMO_PARAMS.DemParticleRadius);
    MSMPackPtr sp6 = std::make_shared<MSMPack>(MSM_T4, SPH_DEM_DEMO_PARAMS.DemParticleRadius);

    std::vector<MSMPackPtr> spArray;
    spArray.push_back(sp1);
    spArray.push_back(sp2);
    spArray.push_back(sp3);
    // spArray.emplace_back(sp4);
    // spArray.emplace_back(sp5);
    // spArray.emplace_back(sp6);

    std::vector<float> spArrayProb;
    spArrayProb.push_back(0.2f);
    spArrayProb.push_back(0.4f);
    spArrayProb.push_back(0.4f);

    std::vector<float> radiusRange;
    radiusRange.push_back(SPH_DEM_DEMO_PARAMS.DemParticleRadius / 2.f);
    radiusRange.push_back(SPH_DEM_DEMO_PARAMS.DemParticleRadius);
    radiusRange.push_back(SPH_DEM_DEMO_PARAMS.DemParticleRadius * 1.5f);

    std::vector<float> radiusRangeProb;
    radiusRangeProb.push_back(0.8f);
    radiusRangeProb.push_back(0.2f);
    std::vector<DEMClump> demClumps = mSampler->GetCDFClumpCloudSampling(SPH_DEM_DEMO_PARAMS.DemInitBoxLowestPoint, SPH_DEM_DEMO_PARAMS.DemInitBoxHighestPoint, spArray, spArrayProb, radiusRange, radiusRangeProb);
    SPH_DEM_DEMO_PARAMS.NumOfParticleGroups = demClumps.size();

    // sph particles
    int3 nn = SPH_DEM_DEMO_PARAMS.SphInitBoxSize;
    float3 lower = SPH_DEM_DEMO_PARAMS.SphInitBoxLowestPoint;
    float radius = SPH_DEM_DEMO_PARAMS.SphParticleRadius;
    float offset = 2.f * radius;
    float3 start_point = lower + make_float3(radius);

    for (size_t i = 0; i < nn.x; i++)
    {
        for (size_t j = 0; j < nn.y; j++)
        {
            for (size_t k = 0; k < nn.z; k++)
            {
                float4 p = make_float4(start_point.x + i * offset, start_point.y + j * offset, start_point.z + k * offset, SPH_DEM_DEMO_PARAMS.SphParticleRadius);

                data_block->id().emplace_back(counter++);
                data_block->label().emplace_back(0);
                data_block->positions().emplace_back(p);
                data_block->fpositions().emplace_back(p);
                data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.SphInitBoxColor, 0.f));
                data_block->masses().emplace_back(SPH_DEM_DEMO_PARAMS.SphRestMass);
                data_block->inertia().emplace_back(0.f);

                DEMClumpInfo info;
                data_block->particle_clump_info().emplace_back(info);
            }
        }
    }

    // re-calc sph rest density
    KernelPoly6 poly6_kernel = KernelPoly6(SPH_DEM_DEMO_PARAMS.SphKernelRadius);
    float l = 2 * SPH_DEM_DEMO_PARAMS.SphParticleRadius;
    int n = (int)std::ceil(SPH_DEM_DEMO_PARAMS.SphKernelRadius / l) + 1;

    float r0 = 0.f;
    for (int x = -n; x <= n; ++x)
    {
        for (int y = -n; y <= n; ++y)
        {
            for (int z = -n; z <= n; ++z)
            {
                float3 rij = make_float3(x * l, y * l, z * l);
                r0 += SPH_DEM_DEMO_PARAMS.SphRestMass * poly6_kernel(length(rij));
            }
        }
    }
    SPH_DEM_DEMO_PARAMS.SphRestDensity = r0;

    // dem particles
    if (SPH_DEM_DEMO_PARAMS.DemInitEnable)
    {
        // dem particles
        for (int i = 0; i < SPH_DEM_DEMO_PARAMS.NumOfParticleGroups; i++)
        {
            int subNum = demClumps[i].subNum;
            uint rigidBodyId = demClumps[i].clumpId;
            for (int j = 0; j < subNum; j++)
            {
                data_block->id().emplace_back(counter++);
                data_block->label().emplace_back(1);

                float3 subPos = demClumps[i].centroid + rotate_vector_by_quaternion(demClumps[i].subPos[j], demClumps[i].ori);
                quaternion subOri = cross(demClumps[i].ori, demClumps[i].subOri[j]);

                float radius = demClumps[i].subRadius[j];
                data_block->positions().emplace_back(make_float4(subPos, radius));
                data_block->fpositions().emplace_back(make_float4(subPos, radius));

                float3 subColor = demClumps[i].subColor[j];
                data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.DemInitBoxColor, 0.5f));
                float mass = SPH_DEM_DEMO_PARAMS.DemRestDensity * ((4.f / 3.f) * M_PI * std::powf(radius, 3.f));

                float inertia = 2.f / 5.f * mass * radius * radius;
                data_block->masses().emplace_back(mass);
                data_block->inertia().emplace_back(inertia);

                DEMClumpInfo info;
                info.clumpId = rigidBodyId;
                info.subId = j;
                info.relPos = demClumps[i].subPos[j];
                info.relOri = demClumps[i].subOri[j];
                data_block->particle_clump_info().emplace_back(info);
            }
            data_block->particle_groups().emplace_back(demClumps[i]);
        }
    }

    SPH_DEM_DEMO_PARAMS.NumOfParticles = counter;

    // boundary
    auto boundaries = mSampler->GetBoxSampling(SPH_DEM_DEMO_PARAMS.LowestPoint, SPH_DEM_DEMO_PARAMS.HighestPoint, SPH_DEM_DEMO_PARAMS.BoundaryParticleRadius * 2.f);
    for (size_t i = 0; i < boundaries.size(); i++)
    {
        data_block->bpositions().emplace_back(boundaries[i]);
    }

    for (size_t i = 0; i < BoundaryData.size(); i++)
    {
        data_block->bpositions().emplace_back(BoundaryData[i]);
    }

    SPH_DEM_DEMO_PARAMS.NumOfBoundaries = boundaries.size() + BoundaryData.size();

    return data_block;
}

SphDemDataPtr VolumeEmitterSphDem::EmitShapeSampling(std::vector<float4> Data, std::vector<float3> BoundaryData)
{
    mSampler = std::make_shared<ParticlesSamplerBasic>();
    SphDemDataPtr data_block = std::make_shared<SphDemData>();

    uint counter = 0;
    SPH_DEM_DEMO_PARAMS.NumOfPhases = 2;

    // sph particles
    int3 nn = SPH_DEM_DEMO_PARAMS.SphInitBoxSize;
    float3 lower = SPH_DEM_DEMO_PARAMS.SphInitBoxLowestPoint;
    float sph_radius = SPH_DEM_DEMO_PARAMS.SphParticleRadius;
    float offset = 2.f * sph_radius;
    float3 start_point = lower + make_float3(sph_radius);

    for (size_t i = 0; i < nn.x; i++)
    {
        for (size_t j = 0; j < nn.y; j++)
        {
            for (size_t k = 0; k < nn.z; k++)
            {
                float4 p = make_float4(start_point.x + i * offset, start_point.y + j * offset, start_point.z + k * offset, SPH_DEM_DEMO_PARAMS.SphParticleRadius);

                data_block->id().emplace_back(counter++);
                data_block->label().emplace_back(0);
                data_block->positions().emplace_back(p);

                if (SPH_DEM_DEMO_PARAMS.SPHVisuable)
                    data_block->fpositions().emplace_back(p);
                else
                    data_block->fpositions().emplace_back(make_float4(p.x, p.y, p.z, 0.f));

                data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.SphInitBoxColor, 0.f));

                data_block->masses().emplace_back(SPH_DEM_DEMO_PARAMS.SphRestMass);
                data_block->inertia().emplace_back(0.f);
            }
        }
    }

    // re-calc sph rest density
    auto poly6_kernel = KernelPoly6(SPH_DEM_DEMO_PARAMS.SphKernelRadius);
    float l = 2 * SPH_DEM_DEMO_PARAMS.SphParticleRadius;
    int n = (int)std::ceil(SPH_DEM_DEMO_PARAMS.SphKernelRadius / l) + 1;

    float r0 = 0.f;
    for (int x = -n; x <= n; ++x)
    {
        for (int y = -n; y <= n; ++y)
        {
            for (int z = -n; z <= n; ++z)
            {
                float3 rij = make_float3(x * l, y * l, z * l);
                r0 += SPH_DEM_DEMO_PARAMS.SphRestMass * poly6_kernel(length(rij));
            }
        }
    }
    SPH_DEM_DEMO_PARAMS.SphRestDensity = r0;

    // define random engine
    float3 color1 = make_float3(0.4f, 0.32f, 0.2f), color2 = make_float3(0.72f, 0.706f, 0.405f), color3 = make_float3(1.f);

    std::array<float, 3> inter_vals = {0.f, 0.9f, 1.f};
    std::array<float, 2> interpolate_vals = {0.9f, 0.1f};

    std::random_device engine;
    std::mt19937 gen(engine());
    std::piecewise_constant_distribution<> pcdis(inter_vals.begin(), inter_vals.end(), interpolate_vals.begin());

    if (SPH_DEM_DEMO_PARAMS.DemInitEnable)
    {

        float shape_particle_miny = SPH_DEM_DEMO_PARAMS.HighestPoint.y;
        float shape_particle_min_radius = SPH_DEM_DEMO_PARAMS.DemParticleRadius;
        for (size_t i = 0; i < Data.size(); i++)
        {
            if (Data[i].y < shape_particle_miny)
                shape_particle_miny = Data[i].y;

            if (Data[i].w < shape_particle_min_radius)
                shape_particle_min_radius = Data[i].w;
        }

        SPH_DEM_DEMO_PARAMS.DemParticleRadius = shape_particle_min_radius;
        float shape_offsety = SPH_DEM_DEMO_PARAMS.LowestPoint.y + shape_particle_min_radius - shape_particle_miny;
        float3 shape_offset = SPH_DEM_DEMO_PARAMS.ShapeSamplingOffset;

        if (SPH_DEM_DEMO_PARAMS.ShapeSamplingOffsetForce)
            shape_offset.y = shape_offsety;

        for (int i = 0; i < Data.size(); i++)
        {
            float3 pos = make_float3(Data[i].x, Data[i].y, Data[i].z) + shape_offset;
            float radius = Data[i].w;

            data_block->id().emplace_back(counter++);
            data_block->label().emplace_back(1);

            data_block->positions().emplace_back(make_float4(pos.x, pos.y, pos.z, radius));
            data_block->fpositions().emplace_back(make_float4(pos.x, pos.y, pos.z, radius));
            //data_block->fpositions().emplace_back(make_float4(pos.x, pos.y, pos.z, 0.f));

            if (SPH_DEM_DEMO_PARAMS.ShapeSamplingRampColorEnable)
            {
                float weight = pcdis(gen);
                if (weight < 0.9f)
                {
                    data_block->colors().emplace_back(make_float4(linear_ramp(color1, color2, weight / 0.9f), 0.5f));
                }
                else
                {
                    data_block->colors().emplace_back(make_float4(linear_ramp(color2, color3, (weight - 0.9f) / 0.1f), 0.5f));
                }
            }
            else
            {
                data_block->colors().emplace_back(make_float4(SPH_DEM_DEMO_PARAMS.DemInitBoxColor, 0.5f));
            }

            float mass = SPH_DEM_DEMO_PARAMS.DemRestDensity * ((4.f / 3.f) * M_PI * std::powf(radius, 3.f));
            float inertia = 2.f / 5.f * mass * radius * radius;
            data_block->masses().emplace_back(mass);
            data_block->inertia().emplace_back(inertia);
        }
    }

    SPH_DEM_DEMO_PARAMS.NumOfParticles = counter;
    //KIRI_LOG_DEBUG("Total Number={0:d}", counter);

    // boundary
    auto boundaries = mSampler->GetBoxSampling(SPH_DEM_DEMO_PARAMS.LowestPoint, SPH_DEM_DEMO_PARAMS.HighestPoint, SPH_DEM_DEMO_PARAMS.BoundaryParticleRadius * 2.f);
    for (size_t i = 0; i < boundaries.size(); i++)
    {
        data_block->bpositions().emplace_back(boundaries[i]);
    }

    for (size_t i = 0; i < BoundaryData.size(); i++)
    {
        data_block->bpositions().emplace_back(BoundaryData[i]);
    }

    SPH_DEM_DEMO_PARAMS.NumOfBoundaries = boundaries.size() + BoundaryData.size();

    return data_block;
}