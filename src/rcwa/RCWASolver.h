#pragma once
#include "defns.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <vector>
#include <map>
#include <memory>

class Layer;

class BlockToeplitzMatrixXcs;

class RCWASolver
{
public:
	int nx_; // number of different Fx harmonics
	int ny_; // number of different Fy harmonics

	scalar Lx_; // the spatial period along x direction
	scalar Ly_; // the spatial period along y direction

	// PML boundaries
	scalar b0_; // bottom of simulation region
	scalar b1_; // top of bottom PML region
	scalar u0_;
	scalar u1_;

	scalar l0_;
	scalar l1_;
	scalar r0_;
	scalar r1_;

	Eigen::VectorXcs Kx_;
	Eigen::VectorXcs Ky_;

	Eigen::MatrixXcs matKx_;
	Eigen::MatrixXcs matKy_;
	
	bool enablePML_;

	std::vector< std::shared_ptr< Layer > > layers_;
	bool isLayerSolved_;

	std::vector<Eigen::MatrixXcs> interfacesE_;
	std::vector<Eigen::MatrixXcs> interfacesH_;
	std::vector<int> interfaceMap_;
	bool isInterfaceSolved_;

	std::vector<Eigen::MatrixXcs> taus_;


	Eigen::MatrixXcs Tdd_;
	Eigen::MatrixXcs Rud_;
	bool isScatterMatrixSolved_;

	friend class SegmentBase;
	friend class SegmentConverter;
	friend class SegmentSplitter;
	friend class AsymModeConverter;

public:
	RCWASolver(int nX, int nY)
	{
		nx_ = 2 * nX + 1;
		ny_ = 2 * nY + 1;
		Lx_ = -1.0;
		Ly_ = -1.0;

		enablePML_ = true;
		isLayerSolved_ = false;
		isInterfaceSolved_ = false;
		isScatterMatrixSolved_ = false;
	}
	
	~RCWASolver();

	RCWASolver(const RCWASolver& solver);
	
	void addLayer(const Layer& layer);

	int nLayerTypes() const { return layers_.size(); }
	
	std::shared_ptr<const Layer> getLayer(int index) { return layers_[index]; }
	
	void solve(
		scalar lambda,
		const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness,
		const std::vector<scalar>* translation = nullptr);

	void saveModeImage(const std::string& filename,
		int layerType, int modeIndex,
		FieldComponent opt, int nResX, int nResY);

	void saveFieldImage(
		const std::string& filename,
		SliceType sliceType,
		scalar sliceCoord,
		FieldComponent fieldComponent,
		SaveOption opt,
		int inputMode,
		const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness,
		const std::vector<scalar>* translation = nullptr);

	void saveFieldImage(
		const std::string& filename,
		SliceType sliceType,
		scalar sliceCoord,
		FieldComponent fieldComponent,
		SaveOption opt,
		const std::vector<std::pair<int, scalex>>& inputCoeffs,
		const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness,
		const std::vector<scalar>* translation = nullptr);
	
	void saveDeviceImage(
		const std::string& filename,
		SliceType sliceType,
		scalar sliceCoord,
		const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness,
		const std::vector<scalar>* translation = nullptr);
				
	void findGuidedModes(
		std::vector<std::pair<int, scalex>>& guidedModeRecords,
		int layerType,
		scalar k0,
		scalar imagCoeff,
		scalar realCoeffMin,
		scalar realCoeffMax);

	scalex rud(int row, int col) const;
	scalex tdd(int row, int col) const;

    const Eigen::MatrixXcs& Tdd() const { return Tdd_; }
    const Eigen::MatrixXcs& Rud() const { return Rud_; }
    
	scalar transmittance(
		int inputMode,
		int outputMode) const;

	scalar reflectance(
		int inputMode,
		int outputMode) const;

	scalar transmittance(int inputMode) const;
	
	scalar reflectance(int inputMode) const;

	void generateHorizontalPlaneWave(
		scalar px,
		scalar py,
		int layerType,
		Eigen::VectorXcs& c);

	void scatterPlaneWave(
		const Eigen::VectorXcs& cInc,
		int refLayerType,
		int trnLayerType,
		scalar k0,
		Eigen::VectorXs& REF,
		Eigen::VectorXs& TRN);

	bool isLayerSolved() const { return isLayerSolved_; }
	bool isInterfaceSolved() const { return isInterfaceSolved_; }
	bool isScatterMatrixSolved() const { return isScatterMatrixSolved_; }

	void enablePML() { enablePML_ = true; }
	void disablePML() { enablePML_ = false; }
	
    void evaluateKMatrices(Eigen::MatrixXcs& Kx, Eigen::MatrixXcs& Ky);

    int nx() const { return nx_; }
    int ny() const { return ny_; }
private:
	void solveLayers(scalar lambda);

	void solveInterfaces(const std::vector<int>& layerStack,
		const std::vector<scalar>* translation = nullptr);

	void solveScatterMatrix(const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness);
	
	void evaluateKMatrices(
		int nx, int ny);
	
	void evaluateAlphaBeta(
		Eigen::DiagonalMatrixXcs& alpha,
		Eigen::DiagonalMatrixXcs& beta);

	void evaluatePML(
		scalex gamma,
		BlockToeplitzMatrixXcs& Fx,
		BlockToeplitzMatrixXcs& Fy);

	void evaluateScatterMatrix(
		Eigen::MatrixXcs& Tdd,
		Eigen::MatrixXcs& Rud,
		const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness);
	
	void evaluateInterfaces(
		std::vector<Eigen::MatrixXcs>& interfacesE,
		std::vector<Eigen::MatrixXcs>& interfacesH,
		std::vector<int>& interfaceMap,
		const std::vector<int>& layerStack,
		const std::vector<scalar>* translation = nullptr);

	void evaluateIntermediateField(
		std::vector<Eigen::VectorXcs>& c_m,
		std::vector<Eigen::VectorXcs>& c_p,
		int inputMode,
		const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness);
	
	void evaluateIntermediateField(
		std::vector<Eigen::VectorXcs>& c_m,
		std::vector<Eigen::VectorXcs>& c_p,
		const std::vector<std::pair<int, scalex>>& inputCoeffs,
		const std::vector<int>& layerStack,
		const std::vector<scalar>& thickness);
};