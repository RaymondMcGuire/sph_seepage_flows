#include <random>
#include <list>
#include <kiri_pbs_cuda/particle/particles_sampler_basic.h>
#include <Eigen/Eigenvalues>

ParticlesSamplerBasic::ParticlesSamplerBasic()
{
}

std::vector<float3> ParticlesSamplerBasic::GetBoxSampling(float3 lower, float3 upper, float spacing)
{
    mPoints.clear();

    int epsilon = 0;
    float3 sides = (upper - lower) / spacing;

    //ZX plane - bottom
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, lower.y, lower.z + j * spacing));
        }
    }

    //ZX plane - top
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, upper.y, lower.z + j * spacing));
        }
    }

    //XY plane - back
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.y + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, lower.y + j * spacing, lower.z));
        }
    }

    //XY plane - front
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.y - epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, lower.y + j * spacing, upper.z));
        }
    }

    //YZ plane - left
    for (int i = -epsilon; i <= sides.y + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x, lower.y + i * spacing, lower.z + j * spacing));
        }
    }

    //YZ plane - right
    for (int i = -epsilon; i <= sides.y + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(upper.x, lower.y + i * spacing, lower.z + j * spacing));
        }
    }
    return mPoints;
}

std::vector<DEMSphere> ParticlesSamplerBasic::GetCloudSampling(float3 lower, float3 upper, float rMean, float rFuzz, int maxNumOfParticles)
{
    std::vector<DEMSphere> mPack;

    // define random engine
    std::random_device engine;
    std::mt19937 gen(engine());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    float r = 0.f;
    const int maxTry = 20;
    for (int i = 0; (i < maxNumOfParticles) || (maxNumOfParticles < 0); i++)
    {
        float rand;
        if (maxNumOfParticles > 0)
            rand = (maxNumOfParticles - (float)i + 0.5f) / (maxNumOfParticles + 1.f);
        else
            rand = distribution(gen);

        int t;

        r = rMean * (2.f * (rand - 0.5f) * rFuzz + 1.f); // uniform distribution in rMean*(1±rRelFuzz)
        //r = rMean;

        // try to put the sphere into a free spot
        for (t = 0; t < maxTry; ++t)
        {
            float3 diagonal = upper - lower;

            float diaX = diagonal.x != 0.f ? distribution(gen) * (diagonal.x - 2.f * r) + r : 0.f;
            float diaY = diagonal.y != 0.f ? distribution(gen) * (diagonal.y - 2.f * r) + r : 0.f;
            float diaZ = diagonal.z != 0.f ? distribution(gen) * (diagonal.z - 2.f * r) + r : 0.f;

            float3 center = make_float3(diaX, diaY, diaZ) + lower;

            size_t packSize = mPack.size();
            bool overlap = false;

            for (size_t j = 0; j < packSize; j++)
            {
                if (std::powf(mPack[j].radius + r, 2.f) >= lengthSquared(mPack[j].center - center))
                {
                    overlap = true;
                    break;
                }
            }

            if (!overlap)
            {
                mPack.push_back(DEMSphere(center, r));
                break;
            }
        }
        if (t == maxTry)
        {
            return mPack;
        }
    }

    return mPack;
}

std::vector<DEMClump> ParticlesSamplerBasic::GetRndClumpCloudSampling(float3 lower, float3 upper, const std::vector<MSMPackPtr> &clumpTypes, float rMean, float rFuzz, int maxNumOfClumps)
{
    mPack.clear();

    std::vector<DEMClump> _clumpPacks;
    std::vector<MSMPack> clumps;
    std::vector<float> boundRadius;

    float maxRadius = 0.f;

    // define random engine
    std::random_device engine;
    std::mt19937 gen(engine());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (const MSMPackPtr &c : clumpTypes)
    {
        MSMPack c2(*c);

        float rand = distribution(gen);
        float radius = rMean * (2.f * (rand - 0.5f) * rFuzz + 1.f); // uniform distribution in rMean*(1±rRelFuzz)

        c2.updateDefaultTypeRadius(radius);

        // convert pack model coord to world coordinate
        c2.translate(c2.midPoint());
        clumps.push_back(c2);

        float r = 0.f;
        for (const auto &s : c2.pack())
        {
            r = fmaxf(r, length(s.center) + s.radius);
        }

        boundRadius.push_back(r);

        float3 cMin, cMax;
        c2.aabb(cMin, cMax);
        for (const auto &s : c2.pack())
        {
            maxRadius = fmaxf(maxRadius, s.radius);
        }
    }

    // clumps generator
    std::list<DEMClumpForGenerator> genClumps;
    const auto maxTry = 200;
    int numOfClumps = 0; // number of clumps generated
    while (numOfClumps < maxNumOfClumps || maxNumOfClumps < 0)
    {
        int clumpChoice = (int)(distribution(gen) * (clumps.size() - 1e-20f));
        int tries = 0;
        while (true)
        {
            float3 diagonal = upper - lower;
            float3 pos = make_float3(distribution(gen) * diagonal.x, distribution(gen) * diagonal.y, distribution(gen) * diagonal.z) + lower;
            float3 color = make_float3(distribution(gen), distribution(gen), distribution(gen));

            // TODO: check this random orientation is homogeneously distributed
            // Note: I've seen some proofs it is not. Chosing uniformly distributed orientation needs more work / Janek
            quaternion ori = make_quaternion(distribution(gen), distribution(gen), distribution(gen), distribution(gen));
            ori = normalize(ori);
            // copy the packing and rotate
            MSMPack C(clumps[clumpChoice]);
            C.rotateAroundOrigin(ori);
            C.translate(pos);
            C.setColor(color);
            const float &rad(boundRadius[clumpChoice]);

            DEMClumpForGenerator ci; // to be used later, but must be here because of goto's

            // check overlap: box margin
            float3 _upper = fmaxf(pos + rad * ones(), upper);
            float3 _lower = fminf(pos - rad * ones(), lower);
            if (_upper != upper || _lower != lower)
            {
                for (const auto &s : C.pack())
                {
                    float3 _sUpper = fmaxf(s.center + s.radius * ones(), upper);
                    float3 _sLower = fminf(s.center - s.radius * ones(), lower);
                    if (_sUpper != upper || _sLower != lower)
                    {
                        goto overlap;
                    }
                }
            }
            // check overlaps: other clumps
            for (const DEMClumpForGenerator &gclump : genClumps)
            {
                bool detailedCheck = false;
                // check overlaps between individual spheres and bounding sphere of the other clump[

                if (lengthSquared(pos - gclump.center) < std::powf(rad + gclump.radius, 2.f))
                {

                    for (const auto &s : C.pack())
                    {
                        if (std::powf(s.radius + gclump.radius, 2.f) > lengthSquared(s.center - gclump.center))
                        {
                            detailedCheck = true;
                            break;
                        }
                    }
                }
                // check sphere-by-sphere, since bounding spheres did overlap
                if (detailedCheck)
                {
                    for (const auto &s : C.pack())
                    {
                        for (int id = gclump.minId; id <= gclump.maxId; id++)
                        {
                            if (lengthSquared(s.center - mPack[id].center) < std::powf(s.radius + mPack[id].radius, 2.f))
                            {
                                goto overlap;
                            }
                        }
                    }
                }
            }

            ci.clumpId = numOfClumps;
            ci.center = pos;
            ci.radius = rad;
            ci.minId = mPack.size();
            ci.maxId = mPack.size() + C.pack().size() - 1;

            // for clump
            DEMClump clumpPacks;
            float density = 1000.f;
            float clumpMass = 0.f;
            float3 clumpMoment = make_float3(0.f);
            tensor3x3 inertiaTensor = make_tensor3x3(0.f);

            int subNum = C.pack().size();
            for (size_t i = 0; i < subNum; i++)
            {
                //std::cout << "realPos=" << C.pack()[i].center.x << "," << C.pack()[i].center.y << "," << C.pack()[i].center.z << std::endl;
                mPack.push_back(DEMSphere(C.pack()[i].center, C.pack()[i].radius, C.pack()[i].color, ci.clumpId));

                float volume = (4.f / 3.f) * M_PI * std::powf(C.pack()[i].radius, 3.f);
                float mass = density * volume;
                clumpMass += mass;
                clumpMoment += mass * C.pack()[i].center;

                float inertia = 2.f / 5.f * mass * C.pack()[i].radius * C.pack()[i].radius;
                float3 inertiaF3 = make_float3(inertia);
                tensor3x3 itd = make_diagonal(inertiaF3);
                inertiaTensor += inertiaTensorTranslate(itd, mass, -1.f * C.pack()[i].center);
            }
            clumpPacks.clumpId = numOfClumps;
            clumpPacks.centroid = clumpMoment / clumpMass;
            clumpPacks.mass = clumpMass;
            clumpPacks.vel = make_float3(0.f);
            clumpPacks.angVel = make_float3(0.f);
            clumpPacks.angMom = make_float3(0.f);
            clumpPacks.subNum = subNum;

            // this will calculate translation only, since rotation is zero
            tensor3x3 Ic_orientG = inertiaTensorTranslate(
                inertiaTensor, -clumpMass /* negative mass means towards centroid */, clumpPacks.centroid); // inertia at clump's centroid but with world orientation
            tensor3x3 symm_Ic_orientG = make_symmetrize(Ic_orientG);

            Eigen::Matrix3f eigen_symm_Ic_orientG;
            eigen_symm_Ic_orientG << symm_Ic_orientG.e1.x, symm_Ic_orientG.e1.y, symm_Ic_orientG.e1.z,
                symm_Ic_orientG.e2.x, symm_Ic_orientG.e2.y, symm_Ic_orientG.e2.z,
                symm_Ic_orientG.e3.x, symm_Ic_orientG.e3.y, symm_Ic_orientG.e3.z;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_decomposed(eigen_symm_Ic_orientG);
            const Eigen::Matrix3f &R_g2c(eigen_decomposed.eigenvectors());
            Eigen::Quaternionf eigen_ori = Eigen::Quaternionf(R_g2c);
            eigen_ori.normalize();
            Eigen::Vector3f eigen_inertia = eigen_decomposed.eigenvalues();

            clumpPacks.ori = make_quaternion(eigen_ori.w(), make_float3(eigen_ori.x(), eigen_ori.y(), eigen_ori.z()));
            // std::cout << "ori=" << eigen_ori.x() << "," << eigen_ori.y() << "," << eigen_ori.z() << "," << eigen_ori.w() << std::endl;
            clumpPacks.inertia = make_float3(eigen_inertia.x(), eigen_inertia.y(), eigen_inertia.z());

            for (size_t i = 0; i < subNum; i++)
            {
                Eigen::Vector3f eigen_body_pos;
                Eigen::Vector3f eigen_subbody_pos;
                eigen_body_pos << clumpPacks.centroid.x, clumpPacks.centroid.y, clumpPacks.centroid.z;
                eigen_subbody_pos << C.pack()[i].center.x, C.pack()[i].center.y, C.pack()[i].center.z;

                Eigen::Quaternionf eigen_subbody_local_ori = eigen_ori.conjugate() * Eigen::Quaternionf::Identity();
                Eigen::Vector3f eigen_subbody_local_pos = eigen_ori.conjugate() * (eigen_subbody_pos - eigen_body_pos);

                clumpPacks.subColor[i] = C.pack()[i].color;
                clumpPacks.subRadius[i] = C.pack()[i].radius;
                clumpPacks.subPos[i] = make_float3(eigen_subbody_local_pos.x(), eigen_subbody_local_pos.y(), eigen_subbody_local_pos.z());
                clumpPacks.subOri[i] = make_quaternion(eigen_subbody_local_ori.w(),
                                                       make_float3(eigen_subbody_local_ori.x(), eigen_subbody_local_ori.y(), eigen_subbody_local_ori.z()));

                clumpPacks.force[i] = make_float3(0.f);
                clumpPacks.torque[i] = make_float3(0.f);
            }
            _clumpPacks.push_back(clumpPacks);

            genClumps.push_back(ci);
            numOfClumps++;

            break; // break away from the try-loop

        overlap:
            if (tries++ == maxTry)
            {
                //return mPack;
                return _clumpPacks;
            }
        }
    }
    //return mPack;
    return _clumpPacks;
}

std::vector<DEMClump> ParticlesSamplerBasic::GetCDFClumpCloudSampling(float3 lower, float3 upper, const std::vector<MSMPackPtr> clumpTypes, const std::vector<float> clumpTypesProb, const std::vector<float> radiusRange, const std::vector<float> radiusConstantProb, int maxNumOfClumps)
{
    mPack.clear();

    std::vector<DEMClump> _clumpPacks;
    std::vector<MSMPack> clumps;
    std::vector<float> boundRadius;

    float maxRadius = 0.f;

    // define random engine
    std::random_device engine;
    std::mt19937 gen(engine());

    std::piecewise_constant_distribution<float> pcdis{std::begin(radiusRange), std::end(radiusRange), std::begin(radiusConstantProb)};
    int maxClumpSamplingNum = 100;
    for (int i = 0; i < clumpTypes.size(); i++)
    {
        MSMPack c2(*clumpTypes[i]);

        int currentClumpGenNum = maxClumpSamplingNum * clumpTypesProb[i];
        for (int j = 0; j < currentClumpGenNum; j++)
        {
            float radius = pcdis(gen);

            c2.updateDefaultTypeRadius(radius);

            // convert pack model coord to world coordinate
            c2.translate(c2.midPoint());
            clumps.push_back(c2);

            float r = 0.f;
            for (const auto &s : c2.pack())
            {
                r = fmaxf(r, length(s.center) + s.radius);
            }

            boundRadius.push_back(r);

            float3 cMin, cMax;
            c2.aabb(cMin, cMax);
            for (const auto &s : c2.pack())
            {
                maxRadius = fmaxf(maxRadius, s.radius);
            }
        }
    }

    std::uniform_real_distribution<float> udis(0.0, 1.0);

    // clumps generator
    std::list<DEMClumpForGenerator> genClumps;
    const auto maxTry = 200;
    int numOfClumps = 0; // number of clumps generated
    while (numOfClumps < maxNumOfClumps || maxNumOfClumps < 0)
    {
        int clumpChoice = (int)(udis(gen) * (clumps.size() - 1e-20f));
        int tries = 0;
        while (true)
        {
            float3 diagonal = upper - lower;
            float3 pos = make_float3(udis(gen) * diagonal.x, udis(gen) * diagonal.y, udis(gen) * diagonal.z) + lower;
            float3 color = make_float3(udis(gen), udis(gen), udis(gen));

            quaternion ori = make_quaternion(udis(gen), udis(gen), udis(gen), udis(gen));
            ori = normalize(ori);
            // copy the packing and rotate
            MSMPack C(clumps[clumpChoice]);
            C.rotateAroundOrigin(ori);
            C.translate(pos);
            C.setColor(color);
            const float &rad(boundRadius[clumpChoice]);

            DEMClumpForGenerator ci; // to be used later, but must be here because of goto's

            // check overlap: box margin
            float3 _upper = fmaxf(pos + rad * ones(), upper);
            float3 _lower = fminf(pos - rad * ones(), lower);
            if (_upper != upper || _lower != lower)
            {
                for (const auto &s : C.pack())
                {
                    float3 _sUpper = fmaxf(s.center + s.radius * ones(), upper);
                    float3 _sLower = fminf(s.center - s.radius * ones(), lower);
                    if (_sUpper != upper || _sLower != lower)
                    {
                        goto overlap;
                    }
                }
            }
            // check overlaps: other clumps
            for (const DEMClumpForGenerator &gclump : genClumps)
            {
                bool detailedCheck = false;
                // check overlaps between individual spheres and bounding sphere of the other clump[

                if (lengthSquared(pos - gclump.center) < std::powf(rad + gclump.radius, 2.f))
                {

                    for (const auto &s : C.pack())
                    {
                        if (std::powf(s.radius + gclump.radius, 2.f) > lengthSquared(s.center - gclump.center))
                        {
                            detailedCheck = true;
                            break;
                        }
                    }
                }
                // check sphere-by-sphere, since bounding spheres did overlap
                if (detailedCheck)
                {
                    for (const auto &s : C.pack())
                    {
                        for (int id = gclump.minId; id <= gclump.maxId; id++)
                        {
                            if (lengthSquared(s.center - mPack[id].center) < std::powf(s.radius + mPack[id].radius, 2.f))
                            {
                                goto overlap;
                            }
                        }
                    }
                }
            }

            ci.clumpId = numOfClumps;
            ci.center = pos;
            ci.radius = rad;
            ci.minId = mPack.size();
            ci.maxId = mPack.size() + C.pack().size() - 1;

            // for clump
            DEMClump clumpPacks;
            float density = 1000.f;
            float clumpMass = 0.f;
            float3 clumpMoment = make_float3(0.f);
            tensor3x3 inertiaTensor = make_tensor3x3(0.f);

            int subNum = C.pack().size();
            for (size_t i = 0; i < subNum; i++)
            {
                //std::cout << "realPos=" << C.pack()[i].center.x << "," << C.pack()[i].center.y << "," << C.pack()[i].center.z << std::endl;
                mPack.push_back(DEMSphere(C.pack()[i].center, C.pack()[i].radius, C.pack()[i].color, ci.clumpId));

                float volume = (4.f / 3.f) * M_PI * std::powf(C.pack()[i].radius, 3.f);
                float mass = density * volume;
                clumpMass += mass;
                clumpMoment += mass * C.pack()[i].center;

                float inertia = 2.f / 5.f * mass * C.pack()[i].radius * C.pack()[i].radius;
                float3 inertiaF3 = make_float3(inertia);
                tensor3x3 itd = make_diagonal(inertiaF3);
                inertiaTensor += inertiaTensorTranslate(itd, mass, -1.f * C.pack()[i].center);
            }
            clumpPacks.clumpId = numOfClumps;
            clumpPacks.centroid = clumpMoment / clumpMass;
            clumpPacks.mass = clumpMass;
            clumpPacks.vel = make_float3(0.f);
            clumpPacks.angVel = make_float3(0.f);
            clumpPacks.angMom = make_float3(0.f);
            clumpPacks.subNum = subNum;

            // this will calculate translation only, since rotation is zero
            tensor3x3 Ic_orientG = inertiaTensorTranslate(
                inertiaTensor, -clumpMass /* negative mass means towards centroid */, clumpPacks.centroid); // inertia at clump's centroid but with world orientation
            tensor3x3 symm_Ic_orientG = make_symmetrize(Ic_orientG);

            Eigen::Matrix3f eigen_symm_Ic_orientG;
            eigen_symm_Ic_orientG << symm_Ic_orientG.e1.x, symm_Ic_orientG.e1.y, symm_Ic_orientG.e1.z,
                symm_Ic_orientG.e2.x, symm_Ic_orientG.e2.y, symm_Ic_orientG.e2.z,
                symm_Ic_orientG.e3.x, symm_Ic_orientG.e3.y, symm_Ic_orientG.e3.z;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_decomposed(eigen_symm_Ic_orientG);
            const Eigen::Matrix3f &R_g2c(eigen_decomposed.eigenvectors());
            Eigen::Quaternionf eigen_ori = Eigen::Quaternionf(R_g2c);
            eigen_ori.normalize();
            Eigen::Vector3f eigen_inertia = eigen_decomposed.eigenvalues();

            clumpPacks.ori = make_quaternion(eigen_ori.w(), make_float3(eigen_ori.x(), eigen_ori.y(), eigen_ori.z()));
            // std::cout << "ori=" << eigen_ori.x() << "," << eigen_ori.y() << "," << eigen_ori.z() << "," << eigen_ori.w() << std::endl;
            clumpPacks.inertia = make_float3(eigen_inertia.x(), eigen_inertia.y(), eigen_inertia.z());

            for (size_t i = 0; i < subNum; i++)
            {
                Eigen::Vector3f eigen_body_pos;
                Eigen::Vector3f eigen_subbody_pos;
                eigen_body_pos << clumpPacks.centroid.x, clumpPacks.centroid.y, clumpPacks.centroid.z;
                eigen_subbody_pos << C.pack()[i].center.x, C.pack()[i].center.y, C.pack()[i].center.z;

                Eigen::Quaternionf eigen_subbody_local_ori = eigen_ori.conjugate() * Eigen::Quaternionf::Identity();
                Eigen::Vector3f eigen_subbody_local_pos = eigen_ori.conjugate() * (eigen_subbody_pos - eigen_body_pos);

                clumpPacks.subColor[i] = C.pack()[i].color;
                clumpPacks.subRadius[i] = C.pack()[i].radius;
                clumpPacks.subPos[i] = make_float3(eigen_subbody_local_pos.x(), eigen_subbody_local_pos.y(), eigen_subbody_local_pos.z());
                clumpPacks.subOri[i] = make_quaternion(eigen_subbody_local_ori.w(),
                                                       make_float3(eigen_subbody_local_ori.x(), eigen_subbody_local_ori.y(), eigen_subbody_local_ori.z()));

                clumpPacks.force[i] = make_float3(0.f);
                clumpPacks.torque[i] = make_float3(0.f);
            }
            _clumpPacks.push_back(clumpPacks);

            genClumps.push_back(ci);
            numOfClumps++;

            break; // break away from the try-loop

        overlap:
            if (tries++ == maxTry)
            {
                //return mPack;
                return _clumpPacks;
            }
        }
    }
    //return mPack;
    return _clumpPacks;
}