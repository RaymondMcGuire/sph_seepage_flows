/*
 * @Author: Xu.Wang 
 * @Date: 2020-05-06 18:44:09 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-06-08 03:05:19
 */

#include <kiri_pbs_cuda/particle/box_particles.h>

void BoxParticles::allocMemory()
{
    mPositions = (float3 *)malloc(mNumOfParticles * sizeof(float3));
    mColors = (float4 *)malloc(mNumOfParticles * sizeof(float4));
    _id = (uint *)malloc(mNumOfParticles * sizeof(uint));
    mLabels = (uint *)malloc(mNumOfParticles * sizeof(uint));
}

BoxParticles::BoxParticles(float3 lower, float3 upper, int3 nn, int type)
{
    mNumOfParticles = nn.x * nn.y * nn.z;
    float3 diff = upper - lower;
    float3 offset3 = make_float3(diff.x / nn.x, diff.y / nn.y, diff.z / nn.z);
    float3 radius3 = offset3 / 2.0f;

    allocMemory();

    // box particles type(0=default cube,1=random cube)
    switch (type)
    {
    case 0:
        for (size_t i = 0; i < nn.x; i++)
        {
            for (size_t j = 0; j < nn.y; j++)
            {
                for (size_t k = 0; k < nn.z; k++)
                {
                    float3 p = make_float3(lower.x + radius3.x + i * offset3.x, lower.y + radius3.y + j * offset3.y, lower.z + radius3.z + k * offset3.z);
                    mPositions[i * nn.y * nn.z + j * nn.z + k] = p;
                    _id[i * nn.y * nn.z + j * nn.z + k] = i * nn.y * nn.z + j * nn.z + k;
                }
            }
        }
        break;
    case 1:
        srand(27);

        uint m_count = 0;

        float sx = lower.x + radius3.x, sy = lower.y + radius3.y, sz = lower.z + radius3.z, x, y, z;

        x = sx;
        for (int i = 0; i < nn.x; i++, x += offset3.x)
        {
            y = sy;
            for (int j = 0; j < nn.y; j++, y += offset3.y)
            {
                z = sz;
                for (int k = 0; k < nn.z; k++, z += offset3.z, m_count++)
                {
                    float r1 = 1.f * rand() / RAND_MAX, r2 = 1.f * rand() / RAND_MAX, r3 = 1.f * rand() / RAND_MAX;
                    mPositions[m_count] = make_float3(x, y, z) + 0.1f * make_float3(sx * r1, sy * r2, sz * r3);
                    _id[m_count] = m_count;
                }
            }
        }
    }
}

BoxParticles::BoxParticles(thrust::host_vector<float3> boxes_lower, thrust::host_vector<float3> boxes_upper, thrust::host_vector<int3> boxes_size, thrust::host_vector<float3> boxes_color)
{
    mNumOfParticles = 0;
    for (int i = 0; i < boxes_size.size(); i++)
    {
        mNumOfParticles += boxes_size[i].x * boxes_size[i].y * boxes_size[i].z;
    }

    allocMemory();

    int counter = 0;
    for (int n = 0; n < boxes_size.size(); n++)
    {
        float3 upper = boxes_upper[n];
        float3 lower = boxes_lower[n];
        int3 nn = boxes_size[n];

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
                    mPositions[counter] = p;
                    _id[counter] = counter;
                    mLabels[counter] = n + 1;
                    mColors[counter] = make_float4(boxes_color[n], 0.f);
                    counter++;
                }
            }
        }
    }
}

BoxParticles::~BoxParticles()
{
    free(mPositions);
    free(_id);
    free(mColors);
    free(mLabels);
}
