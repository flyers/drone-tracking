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

#include <vector>
#include <cassert>
#include <cfloat>
#include "LaRank.h"
#include "Kernels.h"
#include "Rect.h"
#include "Config.h"
#include "mex.h"

static const int kTileSize = 30;

using namespace std;

static const int kMaxSVs = 2000; // TODO (only used when no budget)


LaRank::LaRank(const Config& conf, const Kernel& kernel) :
	m_config(conf),
	m_kernel(kernel),
	m_C(conf.svmC)
{
	int N = conf.svmBudgetSize > 0 ? conf.svmBudgetSize+2 : kMaxSVs;
	m_K.resize(N);
	for (int i = 0; i < N; i++){
		m_K[i].resize(N);
	}
}

LaRank::~LaRank()
{
}

double LaRank::Evaluate(const vector<double>& x, const FloatRect& y) const
{
	double f = 0.0;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		const SupportVector& sv = *m_svs[i];
		f += sv.b * m_kernel.Eval(x, sv.x->x[sv.y]);
	}
	return f;
}

void LaRank::Eval(const vector<vector<double> >& features, const vector<FloatRect>& rects, double* results)
{
	const FloatRect& centre = rects[0];
	for (int i = 0; i < (int)features.size(); ++i)
	{
		// express y in coord frame of centre sample
		FloatRect y = rects[i];
		y.Translate(-centre.XMin(), -centre.YMin());
		results[i] = Evaluate(features[i], y);
	}
}

void LaRank::Update(const vector<vector<double> >& features, const vector<FloatRect>& rects, int y)
{
	// add new support pattern
	SupportPattern* sp = new SupportPattern;
	FloatRect centre = rects[y];
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		// express r in coord frame of centre sample
		FloatRect r = rects[i];
		r.Translate(-centre.XMin(), -centre.YMin());
		sp->yv.push_back(r);
	}
	// evaluate features for each sample
	sp->x = features;
	sp->y = y;
	sp->refCount = 0;
	m_sps.push_back(sp);

	ProcessNew((int)m_sps.size()-1);
	BudgetMaintenance();
	
	for (int i = 0; i < 10; ++i)
	{
		Reprocess();
		BudgetMaintenance();
	}
}

void LaRank::BudgetMaintenance()
{
	if (m_config.svmBudgetSize > 0)
	{
		while ((int)m_svs.size() > m_config.svmBudgetSize)
		{
			BudgetMaintenanceRemove();
		}
	}
}

void LaRank::Reprocess()
{
	ProcessOld();
	for (int i = 0; i < 10; ++i)
	{
		Optimize();
	}
}

double LaRank::ComputeDual() const
{
	double d = 0.0;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		const SupportVector* sv = m_svs[i];
		d -= sv->b*Loss(sv->x->yv[sv->y], sv->x->yv[sv->x->y]);
		for (int j = 0; j < (int)m_svs.size(); ++j)
		{
			d -= 0.5*sv->b*m_svs[j]->b*m_K[i][j];
		}
	}
	return d;
}

void LaRank::SMOStep(int ipos, int ineg)
{
	if (ipos == ineg) return;

	SupportVector* svp = m_svs[ipos];
	SupportVector* svn = m_svs[ineg];
	//assert(svp->x == svn->x);
	SupportPattern* sp = svp->x;

#if VERBOSE
	cout << "SMO: gpos:" << svp->g << " gneg:" << svn->g << endl;
#endif	
	if ((svp->g - svn->g) < 1e-5)
	{
#if VERBOSE
		cout << "SMO: skipping" << endl;
#endif		
	}
	else
	{
		double kii = m_K[ipos][ipos] + m_K[ineg][ineg] - 2*m_K[ipos][ineg];
		double lu = (svp->g-svn->g)/kii;
		// no need to clamp against 0 since we'd have skipped in that case
		double l = min(lu, m_C*(int)(svp->y == sp->y) - svp->b);

		svp->b += l;
		svn->b -= l;

		// update gradients
		for (int i = 0; i < (int)m_svs.size(); ++i)
		{
			SupportVector* svi = m_svs[i];
			svi->g -= l*(m_K[i][ipos] - m_K[i][ineg]);
		}
#if VERBOSE
		cout << "SMO: " << ipos << "," << ineg << " -- " << svp->b << "," << svn->b << " (" << l << ")" << endl;
#endif		
	}
	
	// check if we should remove either sv now
	
	if (fabs(svp->b) < 1e-8)
	{
		RemoveSupportVector(ipos);
		if (ineg == (int)m_svs.size())
		{
			// ineg and ipos will have been swapped during sv removal
			ineg = ipos;
		}
	}

	if (fabs(svn->b) < 1e-8)
	{
		RemoveSupportVector(ineg);
	}
}

pair<int, double> LaRank::MinGradient(int ind)
{
	const SupportPattern* sp = m_sps[ind];
	pair<int, double> minGrad(-1, DBL_MAX);
	for (int i = 0; i < (int)sp->yv.size(); ++i)
	{
		double grad = -Loss(sp->yv[i], sp->yv[sp->y]) - Evaluate(sp->x[i], sp->yv[i]);
		if (grad < minGrad.second)
		{
			minGrad.first = i;
			minGrad.second = grad;
		}
	}
	return minGrad;
}

void LaRank::ProcessNew(int ind)
{
	// gradient is -f(x,y) since loss=0
	int ip = AddSupportVector(m_sps[ind], m_sps[ind]->y, -Evaluate(m_sps[ind]->x[m_sps[ind]->y],m_sps[ind]->yv[m_sps[ind]->y]));

	pair<int, double> minGrad = MinGradient(ind);
	int in = AddSupportVector(m_sps[ind], minGrad.first, minGrad.second);

	SMOStep(ip, in);
}

void LaRank::ProcessOld()
{
	if (m_sps.size() == 0) return;

	// choose pattern to process
	int ind = rand() % m_sps.size();

	// find existing sv with largest grad and nonzero beta
	int ip = -1;
	double maxGrad = -DBL_MAX;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		const SupportVector* svi = m_svs[i];
		if (svi->g > maxGrad && svi->b < m_C*(int)(svi->y == m_sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
	}
	assert(ip != -1);
	if (ip == -1) return;

	// find potentially new sv with smallest grad
	pair<int, double> minGrad = MinGradient(ind);
	int in = -1;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		if (m_svs[i]->y == minGrad.first)
		{
			in = i;
			break;
		}
	}
	if (in == -1)
	{
		// add new sv
		in = AddSupportVector(m_sps[ind], minGrad.first, minGrad.second);
	}

	SMOStep(ip, in);
}

void LaRank::Optimize()
{
	if (m_sps.size() == 0) return;
	
	// choose pattern to optimize
	int ind = rand() % m_sps.size();

	int ip = -1;
	int in = -1;
	double maxGrad = -DBL_MAX;
	double minGrad = DBL_MAX;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		const SupportVector* svi = m_svs[i];
		if (svi->g > maxGrad && svi->b < m_C*(int)(svi->y == m_sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
		if (svi->g < minGrad)
		{
			in = i;
			minGrad = svi->g;
		}
	}
	assert(ip != -1 && in != -1);
	if (ip == -1 || in == -1)
	{
		// this shouldn't happen
		cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		return;
	}

	SMOStep(ip, in);
}

int LaRank::AddSupportVector(SupportPattern* x, int y, double g)
{
	SupportVector* sv = new SupportVector;
	sv->b = 0.0;
	sv->x = x;
	sv->y = y;
	sv->g = g;

	int ind = (int)m_svs.size();
	m_svs.push_back(sv);
	x->refCount++;

#if VERBOSE
	cout << "Adding SV: " << ind << endl;
#endif

	// update kernel matrix
	for (int i = 0; i < ind; ++i)
	{
		m_K[i][ind] = m_kernel.Eval(m_svs[i]->x->x[m_svs[i]->y], x->x[y]);
		m_K[ind][i] = m_K[i][ind];
	}
	m_K[ind][ind] = m_kernel.Eval(x->x[y]);

	return ind;
}

void LaRank::SwapSupportVectors(int ind1, int ind2)
{
	SupportVector* tmp = m_svs[ind1];
	m_svs[ind1] = m_svs[ind2];
	m_svs[ind2] = tmp;
	
	// Swap rows
	for (int i = 0; i < m_K[0].size(); i++){
		double temp = m_K[ind1][i];
		m_K[ind1][i] = m_K[ind2][i];
		m_K[ind2][i] = temp;
	}
	
	// Swap collums
	for (int i = 0; i < m_K.size(); i++){
		double temp = m_K[i][ind1];
		m_K[i][ind1] = m_K[i][ind2];
		m_K[i][ind2] = temp;
	}
}

void LaRank::RemoveSupportVector(int ind)
{
#if VERBOSE
	cout << "Removing SV: " << ind << endl;
#endif	

	m_svs[ind]->x->refCount--;
	if (m_svs[ind]->x->refCount == 0)
	{
		// also remove the support pattern
		for (int i = 0; i < (int)m_sps.size(); ++i)
		{
			if (m_sps[i] == m_svs[ind]->x)
			{
				delete m_sps[i];
				m_sps.erase(m_sps.begin()+i);
				break;
			}
		}
	}

	// make sure the support vector is at the back, this
	// lets us keep the kernel matrix cached and valid
	if (ind < (int)m_svs.size()-1)
	{
		SwapSupportVectors(ind, (int)m_svs.size()-1);
		ind = (int)m_svs.size()-1;
	}
	delete m_svs[ind];
	m_svs.pop_back();
}

void LaRank::BudgetMaintenanceRemove()
{
	// find negative sv with smallest effect on discriminant function if removed
	double minVal = DBL_MAX;
	int in = -1;
	int ip = -1;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->b < 0.0)
		{
			// find corresponding positive sv
			int j = -1;
			for (int k = 0; k < (int)m_svs.size(); ++k)
			{
				if (m_svs[k]->b > 0.0 && m_svs[k]->x == m_svs[i]->x)
				{
					j = k;
					break;
				}
			}
			double val = m_svs[i]->b*m_svs[i]->b*(m_K[i][i] + m_K[j][j] - 2.0*m_K[i][j]);
			if (val < minVal)
			{
				minVal = val;
				in = i;
				ip = j;
			}
		}
	}

	// adjust weight of positive sv to compensate for removal of negative
	m_svs[ip]->b += m_svs[in]->b;

	// remove negative sv
	RemoveSupportVector(in);
	if (ip == (int)m_svs.size())
	{
		// ip and in will have been swapped during support vector removal
		ip = in;
	}
	
	if (m_svs[ip]->b < 1e-8)
	{
		// also remove positive sv
		RemoveSupportVector(ip);
	}

	// update gradients
	// TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		SupportVector& svi = *m_svs[i];
		svi.g = -Loss(svi.x->yv[svi.y],svi.x->yv[svi.x->y]) - Evaluate(svi.x->x[svi.y], svi.x->yv[svi.y]);
	}	
}


