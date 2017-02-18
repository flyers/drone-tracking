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

#ifndef KERNELS_H
#define KERNELS_H

#include <cmath>
#include <vector>
using namespace std;

class Kernel
{
public:
	virtual double Eval(const vector<double>& x1, const vector<double>& x2) const = 0;
	virtual double Eval(const vector<double>& x) const = 0;
};

class LinearKernel : public Kernel
{
public:
	inline double Eval(const vector<double>& x1, const vector<double>& x2) const
	{
		double res = 0.0;
		for (int i = 0; i < x1.size(); i++){
			res += x1[i] * x2[i];
		}
		return res;
	}
	
	inline double Eval(const vector<double>& x) const
	{
		double res = Eval(x, x);
		return res;
	}
};

class GaussianKernel : public Kernel
{
public:
	GaussianKernel(double sigma) : m_sigma(sigma) {}
	inline double Eval(const vector<double>& x1, const vector<double>& x2) const
	{
		double norm = 0;
		for (int i = 0; i < x1.size(); i++){
			norm += (x1[i] - x2[i]) * (x1[i] - x2[i]);
		}
		return exp(-m_sigma*norm);
	}
	
	inline double Eval(const vector<double>& x) const
	{
		return 1.0;
	}

private:
	double m_sigma;
};

class IntersectionKernel : public Kernel
{
public:
	inline double Eval(const vector<double>& x1, const vector<double>& x2) const
	{
		double res = 0.0;
		for (int i = 0; i < x1.size(); i++){
			res += x1[i] < x2[i] ? x1[i] : x2[i];
		}
		return res;
	}
	
	inline double Eval(const vector<double>& x) const
	{
		double res = 0;
		for (int i = 0; i < x.size(); i++){
			res += x[i];
		}
		return res;
	}
};

class Chi2Kernel : public Kernel
{
public:
	inline double Eval(const vector<double>& x1, const vector<double>& x2) const
	{
		double result = 0.0;
		for (int i = 0; i < x1.size(); ++i)
		{
			double a = x1[i];
			double b = x2[i];
			result += (a-b)*(a-b)/(0.5*(a+b)+1e-8);
		}
		return 1.0 - result;
	}
	
	inline double Eval(const vector<double>& x) const
	{
		return 1.0;
	}
};

class MultiKernel : public Kernel
{
public:
	MultiKernel(const std::vector<Kernel*>& kernels, const std::vector<int>& featureCounts) :
		m_n(kernels.size()),
		m_norm(1.0/kernels.size()),
		m_kernels(kernels),
		m_counts(featureCounts)
	{
	}
	
	inline double Eval(const vector<double>& x1, const vector<double>& x2) const
	{
		double sum = 0.0;
		int start = 0;
		for (int i = 0; i < m_n; ++i)
		{
			int c = m_counts[i];
			vector<double> x1_segment (x1.begin() + start, x1.begin() + start + c);
			vector<double> x2_segment (x2.begin() + start, x2.begin() + start + c);
			sum += m_norm*m_kernels[i]->Eval(x1_segment, x2_segment);
			start += c;
		}
		return sum;	
	}
	
	inline double Eval(const vector<double>& x) const
	{
		double sum = 0.0;
		int start = 0;
		for (int i = 0; i < m_n; ++i)
		{
			int c = m_counts[i];
			vector<double> x_segment (x.begin() + start, x.begin() + start + c);
			sum += m_norm*m_kernels[i]->Eval(x_segment);
			start += c;
		}
		return sum;	
	}
	
private:
	int m_n;
	double m_norm;
	std::vector<Kernel*> m_kernels;
	std::vector<int> m_counts;	
	
};

#endif
