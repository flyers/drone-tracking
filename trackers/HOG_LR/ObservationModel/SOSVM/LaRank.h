/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef LARANK_H
#define LARANK_H

#include "Rect.h"

#include <vector>

using namespace std;

class Config;
class Kernel;

class LaRank
{
public:
	LaRank(const Config& conf, const Kernel& kernel);
	~LaRank();
	
	virtual void Eval(const vector<vector<double> >& features, const vector<FloatRect>& rects, double* results);
	virtual void Update(const vector<vector<double> >& features, const vector<FloatRect>& rects, int y);
	
private:

	struct SupportPattern
	{
		vector<vector<double> > x;
		vector<FloatRect> yv;
		//vector<cv::Mat> images;
		int y;
		int refCount;
	};

	struct SupportVector
	{
		SupportPattern* x;
		int y;
		double b;
		double g;
		//cv::Mat image;
	};
	
	const Config& m_config;
	const Kernel& m_kernel;
	
	vector<SupportPattern*> m_sps;
	vector<SupportVector*> m_svs;

	double m_C;
	vector<vector<double> > m_K;

	inline double Loss(const FloatRect& y1, const FloatRect& y2) const
	{
		// overlap loss
		return 1.0-y1.Overlap(y2);
		// squared distance loss
		//double dx = y1.XMin()-y2.XMin();
		//double dy = y1.YMin()-y2.YMin();
		//return dx*dx+dy*dy;
	}
	
	double ComputeDual() const;

	void SMOStep(int ipos, int ineg);
	std::pair<int, double> MinGradient(int ind);
	void ProcessNew(int ind);
	void Reprocess();
	void ProcessOld();
	void Optimize();

	int AddSupportVector(SupportPattern* x, int y, double g);
	void RemoveSupportVector(int ind);
	void RemoveSupportVectors(int ind1, int ind2);
	void SwapSupportVectors(int ind1, int ind2);
	
	void BudgetMaintenance();
	void BudgetMaintenanceRemove();

	double Evaluate(const vector<double>& x, const FloatRect& y) const;
	void UpdateDebugImage();
};

#endif
