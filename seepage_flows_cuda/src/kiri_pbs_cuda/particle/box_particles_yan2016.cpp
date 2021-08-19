/*
 * @Author: Xu.Wang 
 * @Date: 2020-06-20 22:25:55 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-06-20 22:25:55 
 */

#include <kiri_pbs_cuda/particle/box_particles_yan2016.h>

void BoxParticlesYan2016::allocMemory()
{
    _id = (uint *)malloc(_params.NumOfParticles * sizeof(uint));
    mLabels = (uint *)malloc(_params.NumOfParticles * sizeof(uint));
    _phaseType = (int *)malloc(_params.NumOfParticles * sizeof(int));

    mPositions = (float3 *)malloc(_params.NumOfParticles * sizeof(float3));
    mColors = (float4 *)malloc(_params.NumOfParticles * sizeof(float4));

    _boxesColor = (float3 *)malloc(_params.NumOfPhases * sizeof(float3));

    mMasses = (float *)malloc(_params.NumOfParticles * sizeof(float));

    _restMass = (float *)malloc(_params.NumOfPhases * sizeof(float));
    _restDensity = (float *)malloc(_params.NumOfPhases * sizeof(float));
    _coefViscosity = (float *)malloc(_params.NumOfPhases * sizeof(float));

    // boundary
    mBoundaryPositions = (float3 *)malloc(_params.NumOfBoundaries * sizeof(float3));
}

BoxParticlesYan2016::BoxParticlesYan2016(Yan2016Params params)
{
    _params = params;

    // generate boundary particles
    sampler = std::make_shared<ParticlesSamplerBasic>();
    thrust::host_vector<float3> boundaries = sampler->GetBoxSampling(_params.LowestPoint, _params.HighestPoint, _params.particleRadius * 2.f);

    _params.NumOfBoundaries = boundaries.size();
    _params.NumOfPhases = _params.boxes_size.size();
    _params.NumOfParticles = 0;
    for (int n = 0; n < _params.NumOfPhases; n++)
    {
        _params.NumOfParticles += _params.boxes_size[n].x * _params.boxes_size[n].y * _params.boxes_size[n].z;
    }
    _params.numOfTotalParticles = _params.NumOfParticles + _params.NumOfBoundaries;

    allocMemory();

    int counter = 0;

    // add fluid
    for (int n = 0; n < _params.boxes_size.size(); n++)
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

                    _id[counter] = counter;
                    mLabels[counter] = n;
                    _phaseType[counter] = _params.phase_type[n];

                    mPositions[counter] = p;
                    mColors[counter] = make_float4(_boxesColor[n], _params.phase_type[n] / 10.f);

                    mMasses[counter] = _restMass[n];

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
}

BoxParticlesYan2016::~BoxParticlesYan2016()
{
    free(_id);
    free(mLabels);

    free(mPositions);
    free(mBoundaryPositions);

    free(mColors);
    free(_boxesColor);

    free(mMasses);

    free(_restMass);
    free(_restDensity);
    free(_coefViscosity);
    free(_phaseType);
}
