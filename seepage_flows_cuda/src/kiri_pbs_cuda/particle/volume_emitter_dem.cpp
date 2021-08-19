/*** 
 * @Author: Xu.WANG
 * @Date: 2020-09-14 10:03:28
 * @LastEditTime: 2020-10-22 20:11:09
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\volume_emitter_dem.cpp
 */

#include <kiri_pbs_cuda/particle/volume_emitter_dem.h>
#include <random>
DEMDataPtr VolumeEmitterDEM::emitSquareVolumeParticles(uint currentParticlesNum, DEMVolumeParticlesParams params)
{
    DEMDataPtr demData = std::make_shared<DEMData>();
    uint newNumOfParticles = params.size.x * params.size.y * params.size.z;

    uint idx = currentParticlesNum;

    float3 upper = params.upper_point;
    float3 lower = params.lower_point;
    int3 nn = params.size;

    float3 diff = upper - lower;
    float3 offset3 = make_float3(diff.x / nn.x, diff.y / nn.y, diff.z / nn.z);
    float3 radius3 = offset3 / 2.0f;

    for (size_t i = 0; i < nn.x; i++)
    {
        for (size_t j = 0; j < nn.y; j++)
        {
            for (size_t k = 0; k < nn.z; k++)
            {
                float jitter = params.particleRadius * 0.01f;
                float3 p = make_float3(lower.x + radius3.x + i * offset3.x + (frand() * 2.0f - 1.f) * jitter, lower.y + radius3.y + j * offset3.y + (frand() * 2.0f - 1.f) * jitter, lower.z + radius3.z + k * offset3.z + (frand() * 2.0f - 1.f) * jitter);

                demData->id().push_back(idx);
                demData->positions().push_back(make_float4(p, params.particleRadius));
                demData->colors().push_back(make_float4(params.color, 1.f));

                idx++;
            }
        }
    }

    return demData;
}

DEMDataPtr VolumeEmitterDEM::emitMultiBoxByUniformRadius()
{

    DEMDataPtr demData = std::make_shared<DEMData>();

    // number of particles
    DEM_DEMO_PARAMS.NumOfParticles = DEM_DEMO_PARAMS.boxes_size[0].x * DEM_DEMO_PARAMS.boxes_size[0].y * DEM_DEMO_PARAMS.boxes_size[0].z;
    DEM_DEMO_PARAMS.NumOfPhases = DEM_DEMO_PARAMS.boxes_size.size();

    return demData;
}

DEMDataPtr VolumeEmitterDEM::EmitShapeSampling(std::vector<float4> data)
{

    std::random_device engine;
    std::mt19937 gen(engine());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    DEMDataPtr demData = std::make_shared<DEMData>();

    DEM_DEMO_PARAMS.NumOfParticles = data.size();
    DEM_DEMO_PARAMS.NumOfPhases = 1;

    _boxesColor[0] = DEM_DEMO_PARAMS.boxes_color[0];
    _restDensity[0] = DEM_DEMO_PARAMS.rest_density[0];

    float3 offset = make_float3(0.f, -0.88f, 0.f);
    // each phase particle num
    for (int i = 0; i < DEM_DEMO_PARAMS.NumOfParticles; i++)
    {
        float3 pos = make_float3(data[i].x, data[i].y, data[i].z) + offset;
        float radius = data[i].w;
        demData->id().push_back(i);
        demData->positions().push_back(make_float4(pos.x, pos.y, pos.z, radius));
        demData->colors().push_back(make_float4(distribution(gen), distribution(gen), distribution(gen), 0.5f));

        float mass = _restDensity[0] * ((4.f / 3.f) * M_PI * std::powf(radius, 3.f));
        float inertia = 2.f / 5.f * mass * radius * radius;
        demData->mass().push_back(mass);
        demData->inertia().push_back(inertia);
    }

    return demData;
}

DEMDataPtr VolumeEmitterDEM::emitSingleBoxByRandomRadius()
{
    DEMDataPtr demData = std::make_shared<DEMData>();

    float3 upper = DEM_DEMO_PARAMS.boxes_upper[0];
    float3 lower = DEM_DEMO_PARAMS.boxes_lower[0];

    sampler = std::make_shared<ParticlesSamplerBasic>();
    std::vector<DEMSphere> spheres = sampler->GetCloudSampling(lower, upper, DEM_DEMO_PARAMS.AvgParticleRadius, 0.5f);

    // number of particles
    DEM_DEMO_PARAMS.NumOfParticles = spheres.size();
    DEM_DEMO_PARAMS.NumOfPhases = DEM_DEMO_PARAMS.boxes_size.size();

    _boxesColor[0] = DEM_DEMO_PARAMS.boxes_color[0];
    _restDensity[0] = DEM_DEMO_PARAMS.rest_density[0];

    // each phase particle num
    for (int i = 0; i < DEM_DEMO_PARAMS.NumOfParticles; i++)
    {
        demData->id().push_back(i);
        demData->positions().push_back(make_float4(spheres[i].center, spheres[i].radius));
        demData->colors().push_back(make_float4(_boxesColor[0], 0.5f));

        float mass = _restDensity[0] * ((4.f / 3.f) * M_PI * std::powf(spheres[i].radius, 3.f));
        float inertia = 2.f / 5.f * mass * spheres[i].radius * spheres[i].radius;
        demData->mass().push_back(mass);
        demData->inertia().push_back(inertia);
    }

    return demData;
}

DEMDataPtr VolumeEmitterDEM::EmitSingleBoxByMSM()
{

    DEMDataPtr demData = std::make_shared<DEMData>();

    float3 upper = DEM_DEMO_PARAMS.boxes_upper[0];
    float3 lower = DEM_DEMO_PARAMS.boxes_lower[0];

    MSMPackPtr sp1 = std::make_shared<MSMPack>(MSM_S1, DEM_DEMO_PARAMS.AvgParticleRadius);
    MSMPackPtr sp2 = std::make_shared<MSMPack>(MSM_L2, DEM_DEMO_PARAMS.AvgParticleRadius);
    MSMPackPtr sp3 = std::make_shared<MSMPack>(MSM_L3, DEM_DEMO_PARAMS.AvgParticleRadius);
    MSMPackPtr sp4 = std::make_shared<MSMPack>(MSM_C8, DEM_DEMO_PARAMS.AvgParticleRadius);
    MSMPackPtr sp5 = std::make_shared<MSMPack>(MSM_M7, DEM_DEMO_PARAMS.AvgParticleRadius);
    MSMPackPtr sp6 = std::make_shared<MSMPack>(MSM_T4, DEM_DEMO_PARAMS.AvgParticleRadius);

    std::vector<MSMPackPtr> spArray;
    spArray.push_back(sp1);
    spArray.push_back(sp2);
    spArray.push_back(sp3);
    // spArray.push_back(sp4);
    // spArray.push_back(sp5);
    // spArray.push_back(sp6);

    std::vector<float> spArrayProb;
    spArrayProb.push_back(0.2f);
    spArrayProb.push_back(0.4f);
    spArrayProb.push_back(0.4f);

    std::vector<float> radiusRange;
    radiusRange.push_back(DEM_DEMO_PARAMS.AvgParticleRadius / 2.f);
    radiusRange.push_back(DEM_DEMO_PARAMS.AvgParticleRadius);
    radiusRange.push_back(DEM_DEMO_PARAMS.AvgParticleRadius * 1.5f);

    std::vector<float> radiusRangeProb;
    radiusRangeProb.push_back(0.8f);
    radiusRangeProb.push_back(0.2f);

    sampler = std::make_shared<ParticlesSamplerBasic>();
    //std::vector<DEMClump> clumps = sampler->GetRndClumpCloudSampling(lower, upper, spArray, DEM_DEMO_PARAMS.AvgParticleRadius, 0.5f);
    std::vector<DEMClump> clumps = sampler->GetCDFClumpCloudSampling(lower, upper, spArray, spArrayProb, radiusRange, radiusRangeProb);

    std::vector<DEMSphere> packs = sampler->GetPack();

    DEM_DEMO_PARAMS.NumOfParticleGroups = clumps.size();

    DEM_DEMO_PARAMS.NumOfParticles = 0;
    for (size_t nc = 0; nc < DEM_DEMO_PARAMS.NumOfParticleGroups; nc++)
    {
        DEM_DEMO_PARAMS.NumOfParticles += clumps[nc].subNum;
    }

    DEM_DEMO_PARAMS.NumOfPhases = DEM_DEMO_PARAMS.boxes_size.size();

    _boxesColor[0] = DEM_DEMO_PARAMS.boxes_color[0];
    _restDensity[0] = DEM_DEMO_PARAMS.rest_density[0];

    int cnt = 0;
    // each phase particle num
    for (int i = 0; i < DEM_DEMO_PARAMS.NumOfParticleGroups; i++)
    {
        int subNum = clumps[i].subNum;
        for (int j = 0; j < subNum; j++)
        {
            demData->id().push_back(cnt);
            float3 subPos = clumps[i].centroid + rotate_vector_by_quaternion(clumps[i].subPos[j], clumps[i].ori);
            quaternion subOri = cross(clumps[i].ori, clumps[i].subOri[j]);

            float radius = clumps[i].subRadius[j];
            demData->positions().push_back(make_float4(subPos, radius));

            float3 subColor = clumps[i].subColor[j];
            demData->colors().push_back(make_float4(subColor.x, subColor.y, subColor.z, 0.5f));
            float mass = _restDensity[0] * ((4.f / 3.f) * M_PI * std::powf(radius, 3.f));

            float inertia = 2.f / 5.f * mass * radius * radius;
            demData->mass().push_back(mass);
            demData->inertia().push_back(inertia);

            // clump
            DEMClumpInfo info;
            info.clumpId = clumps[i].clumpId;
            info.subId = j;
            info.relPos = clumps[i].subPos[j];
            info.relOri = clumps[i].subOri[j];
            demData->particle_clump_info().push_back(info);

            cnt++;
        }
        demData->particle_groups().push_back(clumps[i]);
    }

    return demData;
}