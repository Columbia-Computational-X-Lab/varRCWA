#pragma once

#include "rcwa/defns.h"
#include "gdstk/gdstk.h"
#include "LayerSampler.h"
#include "Material.h"

class GDSIISampler:
    public LayerSampler
{
private:
    gdstk::Library lib_; // assume only one cell
    
    Eigen::Vector3s shift_; 
    scalar lPML_, rPML_, lspace_, rspace_; // width of PML & 
    scalar bPML_, uPML_, bspace_, uspace_; // space between PML and device
    Material permBackground_, permDevice_;
    scalar thickness_;

    scalar minX_, maxX_, minZ_, maxZ_;

    
    Eigen::VectorXs y_;
    Eigen::VectorXi yIndicator_;
public:
    GDSIISampler(scalar lambda, 
        int nx, 
        int ny,
        const std::string& filename,
        const Eigen::Vector3s& shift = {0., 0., 0.}, // x, y, z
        scalar lPML = 0.5, scalar rPML = 0.5, scalar lspace = 1.5, scalar rspace = 1.5,
        scalar bPML = 0.5, scalar uPML = 0.5, scalar bspace = 1.08, scalar uspace = 1.,
        const Material &permBackground = m_SiO2, const Material &permDevice = m_Si,
        scalar thickness = 0.22);
    
    ~GDSIISampler() override { lib_.clear(); }

    scalar minX() const { return minX_; }
    scalar maxX() const { return maxX_; }
    scalar minZ() const { return minZ_; }
    scalar maxZ() const { return maxZ_; }

    void sampleLayout(scalar z,
        Eigen::VectorXs& x,
        Eigen::MatrixXcs& eps);
    
    const Eigen::VectorXs& y() const { return y_; }

    void sample(scalar z, 
        Eigen::MatrixXcs& P,
        Eigen::MatrixXcs& Q) override;
        
    void sampleOnGPU(scalar z,
        acacia::gpu::complex_t *P,
        acacia::gpu::complex_t *Q) override {}
private:
    void assignSimulationRegion() override;
};