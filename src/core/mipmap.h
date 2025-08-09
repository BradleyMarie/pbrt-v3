
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_MIPMAP_H
#define PBRT_CORE_MIPMAP_H

#include <type_traits>

// core/mipmap.h*
#include "parallel.h"
#include "pbrt.h"
#include "spectrum.h"
#include "stats.h"
#include "texture.h"

namespace pbrt {

STAT_COUNTER("Texture/EWA lookups", nEWALookups);
STAT_COUNTER("Texture/Trilinear lookups", nTrilerpLookups);
STAT_MEMORY_COUNTER("Memory/Texture MIP maps", mipMapMemory);

// MIPMap Helper Declarations
enum class ImageWrap { Repeat, Black, Clamp };
struct ResampleWeight {
    int firstTexel;
    Float weight[4];
};

// MIPMap Declarations
template <typename T>
class MIPMap {
  public:
    // MIPMap Public Methods
    MIPMap(const Point2i &resolution, const T *data, bool doTri = false,
           Float maxAniso = 8.f, ImageWrap wrapMode = ImageWrap::Repeat);
    int Width() const { return resolution[0]; }
    int Height() const { return resolution[1]; }
    int Levels() const { return pyramid.size(); }
    const T &Texel(int level, int s, int t) const;
    T Lookup(const Point2f &st, Float width = 0.f) const;
    T Lookup(const Point2f &st, Vector2f dstdx, Vector2f dstdy) const;

  private:
    // MIPMap Private Methods
    std::unique_ptr<ResampleWeight[]> resampleWeights(int oldRes, int newRes) {
        CHECK_GE(newRes, oldRes);
        std::unique_ptr<ResampleWeight[]> wt(new ResampleWeight[newRes]);
        Float filterwidth = 2.f;
        for (int i = 0; i < newRes; ++i) {
            // Compute image resampling weights for _i_th texel
            Float center = (i + .5f) * oldRes / newRes;
            wt[i].firstTexel = std::floor((center - filterwidth) + 0.5f);
            for (int j = 0; j < 4; ++j) {
                Float pos = wt[i].firstTexel + j + .5f;
                wt[i].weight[j] = Lanczos((pos - center) / filterwidth);
            }

            // Normalize filter weights for texel resampling
            Float invSumWts = 1 / (wt[i].weight[0] + wt[i].weight[1] +
                                   wt[i].weight[2] + wt[i].weight[3]);
            for (int j = 0; j < 4; ++j) wt[i].weight[j] *= invSumWts;
        }
        return wt;
    }
    Float clamp(Float v) { return Clamp(v, 0.f, Infinity); }
    RGBSpectrum clamp(const RGBSpectrum &v) { return v.Clamp(0.f, Infinity); }
    SampledSpectrum clamp(const SampledSpectrum &v) {
        return v.Clamp(0.f, Infinity);
    }
    T triangle(int level, const Point2f &st) const;
    T EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const;

    // MIPMap Private Data
    const bool doTrilinear;
    const Float maxAnisotropy;
    const ImageWrap wrapMode;
    Point2i resolution;
    std::vector<std::unique_ptr<BlockedArray<T>>> pyramid;
    static PBRT_CONSTEXPR int WeightLUTSize = 128;
    static Float weightLut[WeightLUTSize];
};

// MIPMap Method Definitions
template <typename T>
MIPMap<T>::MIPMap(const Point2i &res, const T *img, bool doTrilinear,
                  Float maxAnisotropy, ImageWrap wrapMode)
    : doTrilinear(doTrilinear),
      maxAnisotropy(maxAnisotropy),
      wrapMode(wrapMode),
      resolution(res) {
    ProfilePhase _(Prof::MIPMapCreation);

    std::unique_ptr<T[]> resampledImage = nullptr;
    if (!IsPowerOf2(resolution[0]) || !IsPowerOf2(resolution[1])) {
        // Resample image to power-of-two resolution
        Point2i resPow2(RoundUpPow2(resolution[0]), RoundUpPow2(resolution[1]));
        LOG(INFO) << "Resampling MIPMap from " << resolution << " to "
                  << resPow2 << ". Ratio= "
                  << (Float(resPow2.x * resPow2.y) /
                      Float(resolution.x * resolution.y));
        // Resample image in $s$ direction
        std::unique_ptr<ResampleWeight[]> sWeights =
            resampleWeights(resolution[0], resPow2[0]);
        resampledImage.reset(new T[resPow2[0] * resPow2[1]]);

        // Apply _sWeights_ to zoom in $s$ direction
        ParallelFor(
            [&](int t) {
                for (int s = 0; s < resPow2[0]; ++s) {
                    // Compute texel $(s,t)$ in $s$-zoomed image
                    resampledImage[t * resPow2[0] + s] = 0.f;
                    for (int j = 0; j < 4; ++j) {
                        int origS = sWeights[s].firstTexel + j;
                        if (wrapMode == ImageWrap::Repeat)
                            origS = Mod(origS, resolution[0]);
                        else if (wrapMode == ImageWrap::Clamp)
                            origS = Clamp(origS, 0, resolution[0] - 1);
                        if (origS >= 0 && origS < (int)resolution[0])
                            resampledImage[t * resPow2[0] + s] +=
                                sWeights[s].weight[j] *
                                img[t * resolution[0] + origS];
                    }
                }
            },
            resolution[1], 16);

        // Resample image in $t$ direction
        std::unique_ptr<ResampleWeight[]> tWeights =
            resampleWeights(resolution[1], resPow2[1]);
        std::vector<T *> resampleBufs;
        int nThreads = MaxThreadIndex();
        for (int i = 0; i < nThreads; ++i)
            resampleBufs.push_back(new T[resPow2[1]]);
        ParallelFor(
            [&](int s) {
                T *workData = resampleBufs[ThreadIndex];
                for (int t = 0; t < resPow2[1]; ++t) {
                    workData[t] = 0.f;
                    for (int j = 0; j < 4; ++j) {
                        int offset = tWeights[t].firstTexel + j;
                        if (wrapMode == ImageWrap::Repeat)
                            offset = Mod(offset, resolution[1]);
                        else if (wrapMode == ImageWrap::Clamp)
                            offset = Clamp(offset, 0, (int)resolution[1] - 1);
                        if (offset >= 0 && offset < (int)resolution[1])
                            workData[t] +=
                                tWeights[t].weight[j] *
                                resampledImage[offset * resPow2[0] + s];
                    }
                }
                for (int t = 0; t < resPow2[1]; ++t)
                    resampledImage[t * resPow2[0] + s] = clamp(workData[t]);
            },
            resPow2[0], 32);
        for (auto ptr : resampleBufs) delete[] ptr;
        resolution = resPow2;
    }
    // Initialize levels of MIPMap from image
    int nLevels = 1 + Log2Int(std::max(resolution[0], resolution[1]));
    pyramid.resize(nLevels);

    // Initialize most detailed level of MIPMap
    pyramid[0].reset(
        new BlockedArray<T>(resolution[0], resolution[1],
                            resampledImage ? resampledImage.get() : img));
    for (int i = 1; i < nLevels; ++i) {
        // Initialize $i$th MIPMap level from $i-1$st level
        int sRes = std::max(1, pyramid[i - 1]->uSize() / 2);
        int tRes = std::max(1, pyramid[i - 1]->vSize() / 2);
        pyramid[i].reset(new BlockedArray<T>(sRes, tRes));

        // Filter four texels from finer level of pyramid
        ParallelFor(
            [&](int t) {
                for (int s = 0; s < sRes; ++s)
                    (*pyramid[i])(s, t) =
                        .25f * (Texel(i - 1, 2 * s, 2 * t) +
                                Texel(i - 1, 2 * s + 1, 2 * t) +
                                Texel(i - 1, 2 * s, 2 * t + 1) +
                                Texel(i - 1, 2 * s + 1, 2 * t + 1));
            },
            tRes, 16);
    }

    // Initialize EWA filter weights if needed
    if (weightLut[0] == 0.) {
        for (int i = 0; i < WeightLUTSize; ++i) {
            Float alpha = 2;
            Float r2 = Float(i) / Float(WeightLUTSize - 1);
            weightLut[i] = std::exp(-alpha * r2) - std::exp(-alpha);
        }
    }
    mipMapMemory += (4 * resolution[0] * resolution[1] * sizeof(T)) / 3;
}

template <typename T>
const T &MIPMap<T>::Texel(int level, int s, int t) const {
    CHECK_LT(level, pyramid.size());
    const BlockedArray<T> &l = *pyramid[level];
    // Compute texel $(s,t)$ accounting for boundary conditions
    switch (wrapMode) {
    case ImageWrap::Repeat:
        s = Mod(s, l.uSize());
        t = Mod(t, l.vSize());
        break;
    case ImageWrap::Clamp:
        s = Clamp(s, 0, l.uSize() - 1);
        t = Clamp(t, 0, l.vSize() - 1);
        break;
    case ImageWrap::Black: {
        static const T black = 0.f;
        if (s < 0 || s >= (int)l.uSize() || t < 0 || t >= (int)l.vSize())
            return black;
        break;
    }
    }
    return l(s, t);
}

template <typename T>
T MIPMap<T>::Lookup(const Point2f &st, Float width) const {
    ++nTrilerpLookups;
    ProfilePhase p(Prof::TexFiltTrilerp);
    return triangle(0, st);
}

template <typename T>
T MIPMap<T>::triangle(int level, const Point2f &st) const {
    if (std::is_same<T, float>::value) {
        level = Clamp(level, 0, Levels() - 1);
        Float s = st[0] * pyramid[level]->uSize() - 0.5f;
        Float t = st[1] * pyramid[level]->vSize() - 0.5f;
        int s0 = std::floor(s), t0 = std::floor(t);
        Float ds = s - s0, dt = t - t0;
        return (1 - ds) * (1 - dt) * Texel(level, s0, t0) +
            (1 - ds) * dt * Texel(level, s0, t0 + 1) +
            ds * (1 - dt) * Texel(level, s0 + 1, t0) +
            ds * dt * Texel(level, s0 + 1, t0 + 1);
    } else {
        Float s = st[0] * pyramid[level]->uSize();
        Float t = st[1] * pyramid[level]->vSize();
        int s0 = std::floor(s), t0 = std::floor(t);
        return Texel(0, s0, t0);
    }
}

template <typename T>
T MIPMap<T>::Lookup(const Point2f &st, Vector2f dst0, Vector2f dst1) const {
    return triangle(0, st);
}

template <typename T>
T MIPMap<T>::EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const {
    return triangle(0, st);
}

template <typename T>
Float MIPMap<T>::weightLut[WeightLUTSize];

}  // namespace pbrt

#endif  // PBRT_CORE_MIPMAP_H
