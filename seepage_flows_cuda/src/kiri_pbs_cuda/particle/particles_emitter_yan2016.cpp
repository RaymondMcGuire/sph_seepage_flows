/*
 * @Author: Xu.Wang 
 * @Date: 2020-07-24 00:00:23 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-07-24 21:28:19
 */

#include <kiri_pbs_cuda/particle/particles_emitter_yan2016.h>

MultiSphYan2016DataPtr ParticlesEmitterYan2016::emitSquareVolumeParticles(uint currentParticlesNum, SquareVolumeParticlesParams params)
{
    MultiSphYan2016DataPtr yan2016Data = std::make_shared<MultiSphYan2016Data>();
    uint newNumOfParticles = params.size.x * params.size.y * params.size.z;

    uint idx = currentParticlesNum;

    yan2016Data->NumOfParticles = newNumOfParticles;
    yan2016Data->restDensity = params.rest_density;
    yan2016Data->phaseType = params.phase_type;
    yan2016Data->restMass = params.rest_mass;
    yan2016Data->coefViscosity = params.coef_viscosity;
    yan2016Data->defaultColor = params.color;

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
                float3 p = make_float3(lower.x + radius3.x + i * offset3.x, lower.y + radius3.y + j * offset3.y, lower.z + radius3.z + k * offset3.z);

                yan2016Data->id().push_back(idx);
                yan2016Data->labels().push_back(params.phase_number);
                yan2016Data->phaseTypes().push_back(yan2016Data->phaseType);
                yan2016Data->positions().push_back(p);
                yan2016Data->colors().push_back(make_float4(yan2016Data->defaultColor, yan2016Data->phaseType / 10.f));
                yan2016Data->masses().push_back(yan2016Data->restMass);

                idx++;
            }
        }
    }

    return yan2016Data;
}

MultiSphYan2016DataPtr ParticlesEmitterYan2016::addBoxParticles(uint currentParticlesNum, uint currentPhasesNum, BoxParticlesParams params)
{
    MultiSphYan2016DataPtr yan2016Data = std::make_shared<MultiSphYan2016Data>();

    uint idx = currentParticlesNum;

    uint newNumOfParticles = params.boxes_size.x * params.boxes_size.y * params.boxes_size.z;

    yan2016Data->NumOfParticles = newNumOfParticles;
    yan2016Data->restDensity = params.rest_density;
    yan2016Data->phaseType = params.phase_type;
    yan2016Data->restMass = params.rest_mass;
    yan2016Data->coefViscosity = params.coef_viscosity;
    yan2016Data->defaultColor = params.boxes_color;

    float3 upper = params.boxes_upper;
    float3 lower = params.boxes_lower;
    int3 nn = params.boxes_size;

    float3 diff = upper - lower;
    float3 offset3 = make_float3(diff.x / nn.x, diff.y / nn.y, diff.z / nn.z);
    float3 radius3 = offset3 / 2.0f;

    for (size_t i = 0; i < nn.x; i++)
    {
        for (size_t j = 0; j < nn.y; j++)
        {
            for (size_t k = 0; k < nn.z; k++)
            {
                float3 p = make_float3(lower.x + radius3.x + i * offset3.x, lower.y + radius3.y + j * offset3.y, lower.z + radius3.z + k * offset3.z);

                yan2016Data->id().push_back(idx);
                yan2016Data->labels().push_back(currentPhasesNum);
                yan2016Data->phaseTypes().push_back(yan2016Data->phaseType);
                yan2016Data->positions().push_back(p);
                yan2016Data->colors().push_back(make_float4(yan2016Data->defaultColor, yan2016Data->phaseType / 10.f));
                yan2016Data->masses().push_back(yan2016Data->restMass);

                idx++;
            }
        }
    }

    return yan2016Data;
}

MultiSphYan2016DataPtr ParticlesEmitterYan2016::addMultiBoxParticles(Yan2016Params params)
{
    MultiSphYan2016DataPtr yan2016Data = std::make_shared<MultiSphYan2016Data>();

    _params = params;

    // generate boundary particles
    sampler = std::make_shared<ParticlesSamplerBasic>();
    thrust::host_vector<float3> boundaries = sampler->GetBoxSampling(_params.LowestPoint, _params.HighestPoint, _params.particleRadius * 2.f);

    _params.NumOfBoundaries = boundaries.size();
    mBoundaryPositions = (float3 *)malloc(_params.NumOfBoundaries * sizeof(float3));

    _params.NumOfPhases = _params.boxes_size.size();

    _params.NumOfParticles = 0;
    for (int n = 0; n < _params.NumOfPhases; n++)
    {
        _params.NumOfParticles += _params.boxes_size[n].x * _params.boxes_size[n].y * _params.boxes_size[n].z;
    }
    _params.numOfTotalParticles = _params.NumOfParticles + _params.NumOfBoundaries;

    uint counter = 0;
    // add fluid
    for (uint n = 0; n < _params.boxes_size.size(); n++)
    {
        _boxesColor[n] = _params.boxes_color[n];
        _restDensity[n] = _params.rest_density[n];
        _restMass[n] = _params.rest_mass[n];
        _coefViscosity[n] = _params.coef_viscosity[n];

        float3 upper = _params.boxes_upper[n];
        float3 lower = _params.boxes_lower[n];
        int3 nn = _params.boxes_size[n];

        float3 diff = upper - lower;
        float3 offset3 = make_float3(diff.x / nn.x, diff.y / nn.y, diff.z / nn.z);
        float3 radius3 = offset3 / 2.0f;

        for (size_t i = 0; i < nn.x; i++)
        {
            for (size_t j = 0; j < nn.y; j++)
            {
                for (size_t k = 0; k < nn.z; k++)
                {
                    float3 p = make_float3(lower.x + radius3.x + i * offset3.x, lower.y + radius3.y + j * offset3.y, lower.z + radius3.z + k * offset3.z);

                    yan2016Data->id().push_back(counter);
                    yan2016Data->labels().push_back(n);
                    yan2016Data->phaseTypes().push_back(_params.phase_type[n]);
                    yan2016Data->positions().push_back(p);
                    yan2016Data->colors().push_back(make_float4(_boxesColor[n], _params.phase_type[n] / 10.f));
                    yan2016Data->masses().push_back(_restMass[n]);

                    counter++;
                }
            }
        }
    }
    // add boundaries
    for (int i = 0; i < boundaries.size(); i++)
    {
        mBoundaryPositions[i] = boundaries[i];
    }

    return yan2016Data;
}