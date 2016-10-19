/* Mask.cpp
Copyright (c) 2014 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#include "Mask.h"

#include "ImageBuffer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

using namespace std;

// Default constructor.
Mask::Mask()
	: radius(0.)
{
}

const Mask Mask::emptymask;

// Helpers for generating a Mask from an ImageBuffer
namespace {
	// Trace out a pixmap.
	void Trace(const ImageBuffer *image, vector<Point> *raw)
	{
		uint32_t on = 0xFF000000;  // alpha channel
		const uint32_t *begin = image->Pixels();
		
		// Convert the pitch to uint32_ts instead of bytes.
		int pitch = image->Width();
		
		// First, find a non-empty pixel.
		// This points to the current pixel.
		const uint32_t *it = begin;
		// This is where we will store the point:
		Point point;
		
		for(int y = 0, height=image->Height(); y < height; ++y)
			for(int x = 0, width=image->Width(); x < width; ++x)
			{
				// If this pixel is occupied, bail out of both loops.
				if(*it & on)
				{
					point.Set(x, y);
					// Break out of both loops.
					y = height;
					break;
				}
				++it;
			}
		
		// Now "it" points to the first pixel, whose coordinates are in "point".
		// We will step around the outline in these 8 basic directions:
		static const Point step[8] = {
			{0., -1.}, {1., -1.}, {1., 0.}, {1., 1.},
			{0., 1.}, {-1., 1.}, {-1., 0.}, {-1., -1.}};
		const int off[8] = {
			-pitch, -pitch + 1, 1, pitch + 1,
			pitch, pitch - 1, -1, -pitch - 1};
		int d = 0;
		// All points must be less than this,
		const double maxX = image->Width() - .5;
		const double maxY = image->Height() - .5;
		
		// Loop until we come back here.
		begin = it;
		do {
			raw->push_back(point);
			
			Point next;
			int firstD = d;
			while(true)
			{
				next = point + step[d];
				// Use padded comparisons in case errors somehow accumulate and
				// the doubles are no longer canceling out to 0.
#ifdef __SSE2__
				__m128d ge_minus_half = _mm_cmpge_pd(next, _mm_set1_pd(-0.5));
				__m128d lt_max = _mm_cmplt_pd(next, _mm_setr_pd(maxX, maxY));
				__m128d anded = _mm_and_pd(ge_minus_half, lt_max);
				bool inbounds = _mm_movemask_pd(anded) == 0b11;  // all 4 conditions true
#else
				unsigned ge_mh = (next.X() >= -.5) & (next.Y() >= -.5);
				unsigned lt_mh = (next.X() < maxX) & (next.Y() < maxY);
				bool inbounds = ge_mh & lt_mh;
#endif
				if(inbounds)
					if(it[off[d]] & on)
						break;
				
				// Advance to the next direction.
				d = (d + 1) & 7;
				// If this point is alone, bail out.
				if(d == firstD)
					return;
			}
			
			point = next;
			it += off[d];
			// Rotate the direction backward ninety degrees.
			d = (d + 6) & 7;
			
			// Loop until we are back where we started.
		} while(it != begin);
	}
	
	
	void SmoothAndCenter(vector<Point> *raw, Point size)
	{
		// Smooth out the outline by averaging neighboring points.
		Point prev = raw->back();
		for(Point &p : *raw)
		{
			prev += p;
			prev -= size;
			// Since we'll always be using these sprites at 50% scale, do that
			// scaling here.
			prev *= .25;
			swap(prev, p);
		}
	}
	
	
	// Distance from a point to a line, squared.
	double Distance(Point p, Point a, Point b)
	{
		// Convert to a coordinate system where a is the origin.
		p -= a;
		b -= a;
		double length = b.LengthSquared();
		if(length)
		{
			// Find out how far along the line the tangent to p intersects.
			double u = b.Dot(p) / length;
			// If it is beyond one of the endpoints, use that endpoint.
			p -= max(0., min(1., u)) * b;
		}
		return p.LengthSquared();
	}
	
	
	void Simplify(const vector<Point> &p, int first, int last, vector<Point> *result)
	{
		// Find the most divergent point.
		double dmax = 0.;
		int imax = 0;
		
		for(int i = first + 1; true; ++i)
		{
			if(static_cast<unsigned>(i) == p.size())
				i = 0;
			if(i == last)
				break;
			
			double d = Distance(p[i], p[first], p[last]);
			// Enforce symmetry by using y position as a tiebreaker rather than
			// just the order in the list.
			if(d > dmax || (d == dmax && p[i].Y() > p[imax].Y()))
			{
				dmax = d;
				imax = i;
			}
		}
		
		// If the most divergent point is close enough to the outline, stop.
		if(dmax < 1.)
			return;
		
		// Recursively simplify the lines to both sides of that point.
		Simplify(p, first, imax, result);
	
		result->push_back(p[imax]);
	
		Simplify(p, imax, last, result);
	}
	
	
	// Simplify the given outline using the Ramer-Douglas-Peucker algorithm.
	void Simplify(const vector<Point> &raw, vector<Point> *result)
	{
		result->clear();
		
		// The image has been scaled to 50% size, so the raw outline must have
		// vertices every half-pixel. Find all vertices with X coordinates
		// within a quarter-pixel of 0, and of those, select the top-most and
		// bottom-most ones.
		int top = -1;
		int bottom = -1;
		for(int i = 0; static_cast<unsigned>(i) < raw.size(); ++i)
			if(raw[i].X() >= -.25 && raw[i].X() < .25)
			{
				if(top == -1)
					top = bottom = i;
				else if(raw[i].Y() > raw[bottom].Y())
					bottom = i;
				else
					top = i;
			}
		
		// Bail out if we couldn't find top and bottom vertices.
		if(top == bottom)
		{
			// happens for the cloaked ship: no outline points
			//cerr << "couldn't find top and bottom points to simplify mask\n";
			return;
		}

		result->push_back(raw[top]);
		Simplify(raw, top, bottom, result);
		result->push_back(raw[bottom]);
		Simplify(raw, bottom, top, result);
	}
}



// Construct a mask from the alpha channel of an SDL surface. (The surface
// must therefore be a 4-byte RGBA format.)
void Mask::Create(const ImageBuffer *image)
{
	outline_simd.clear();

	vector<Point> outline;  // used to be a class member
	{
	vector<Point> raw;
	Trace(image, &raw);
	
	SmoothAndCenter(&raw, Point(image->Width(), image->Height()));
	
	Simplify(raw, &outline);
	// destroy raw here, before allocating outline_simd
	}

#ifdef USE_NONSIMD_OUTLINE
	this->outline = outline;
#endif

//	radius = Radius(outline);

	if(outline.empty())
	{
		// happens for the cloaked ship: a single transparent pixel
		cerr << "Mask::Create: bailing early on empty outline\n";
		return;
	}

	// copy into a SIMD-friendly layout
	const unsigned vecSize = xy_interleave::vecSize;

	Point repeat = outline.front();
	do { // repeat at least once so outline[i+1] gets the last->first segment for i=last
		outline.emplace_back(repeat);
	} while((outline.size() & (vecSize-1)) != 1);  // pad to a full vector, plus the one we don't take

	unsigned fullVectors = outline.size() & ~(vecSize-1);
	outline_simd.reserve(fullVectors/vecSize);
	float rSquared = 0;
	for(unsigned i=0 ; i<fullVectors ; )
	{
		xy_interleave tmp;
		for(unsigned j=0 ; j<vecSize ; ++j)
		{
			Point curr = outline[i];
			Point next = outline[i+1];  // doesn't go off the end because we stop one before that
			tmp.x[j] = curr.X();
			tmp.y[j] = curr.Y();
//			outline_simd[i/4].x[j]
//			outline_simd[i/4].y[j] = outline[i].Y();
			rSquared = max(rSquared, tmp.x[j]*tmp.x[j] + tmp.y[j]*tmp.y[j]);

			Point delta = next - curr;
			tmp.dx[j] = delta.X();
			tmp.dy[j] = delta.Y();
			++i;  // loop over vector<Point>
		}
#ifdef __SSE2__
// TODO: SIMD rSquared
#else
#endif
		outline_simd.emplace_back(tmp);
	}

/*
	if (outline.size() != fullVectors)
	{
		xy_interleave tmp;
		unsigned i=fullVectors;
		for(unsigned j=0; j<4 ; ++j)
		{
			Point next = prev;  // pad to a full vector width with copies of the last point
			if (i<outline.size())
			{
				next = outline[i++];
			}
			tmp.x[j] = next.X();
			tmp.y[j] = next.Y();

			Point delta = next - prev;
			tmp.dx[j] = delta.X();
			tmp.dy[j] = delta.Y();
			prev = next;
		}
		outline_simd.emplace_back(tmp);
	}
*/
	radius = sqrt(rSquared);
	outline_simd.shrink_to_fit();  // clear doesn't reduce capacity; this does
}




// Check if this mask intersects the given line segment (from sA to vA). If
// it does, return the fraction of the way along the segment where the
// intersection occurs. The sA should be relative to this object's center.
// If this object contains the given point, the return value is 0. If there
// is no collision, the return value is 1.
double Mask::Collide(Point sA, Point vA, Angle facing) const
{
	// Bail out if we're too far away to possibly be touching.
	double distance = sA.Length();
	if(outline_simd.empty() || distance > radius + vA.Length())
		return 1.;
	
	// Rotate into the mask's frame of reference.
	sA = (-facing).Rotate(sA);
	vA = (-facing).Rotate(vA);
	
	// If this point is contained within the mask, a ray drawn out from it will
	// intersect the mask an even number of times. If that ray coincides with an
	// edge, ignore that edge, and count all segments as closed at the start and
	// open at the end to avoid double-counting.
	
	// For simplicity, use a ray pointing straight downwards. A segment then
	// intersects only if its x coordinates span the point's coordinates.
	if(distance <= radius && Contains(sA))
		return 0.;

	// TODO: divide into quadrants, and only check the pieces of the outline in that quad if outline.size() > 16
	double tmp = Intersection(sA, vA);
	if(tmp > 1.0 || tmp <= 0.0)
		cerr << "Mask::Collide badness: tmp outside of 0..1.  tmp= " << tmp << '\n';
	return tmp;
}
// d > r + vl
// d^2 > (r+vl)^2           // distances are known to be non-negative
// d^2 > r^2 + 2*r*vl + vl^2
// d^2 > r^2 + vl^2 + 2*sqrt(r^2*vl^2)
// d^2 - r^2 > vl^2 + 2*sqrt(r^2*vl^2)
// d2  - r*r > vl2  + 2*sqrt(r*r * vl2)    // d^2 and vl^2 are what we start with; distinguish from r^2
// d2  - r*r - vl2  >  2*sqrt(r*r * vl2)
// 0.5*(d2  - r*r - vl2)  > sqrt(r*r * vl2)

// 0.5*(d2  - r*r - vl2)  > r * sqrt(vl2)
// (d2  - r*r - vl2)^2  > (r*r) * (4 * vl2) && (d2  - r*r - vl2) > 0

// vs
//sqrt(d^2) > r + sqrt(vl^2)


// Check whether the mask contains the given point.
bool Mask::Contains(Point point, Angle facing) const
{
	if(outline_simd.empty() || point.OutOfRange(radius))
		return false;
	
	// Rotate into the mask's frame of reference.
	return Contains((-facing).Rotate(point));
}



// Find out how close this mask is to the given point. Again, the mask is
// assumed to be rotated and scaled according to the given unit vector.
bool Mask::WithinRange(Point point, Angle facing, double range) const
{
	// Bail out if the object is too far away to possible be touched.
	if(outline_simd.empty() || point.OutOfRange(range + radius))  // was range < point.Length() - radius
		return false;

	// Rotate into the mask's frame of reference.
	point = (-facing).Rotate(point);

#ifdef __SSE2__
	__m128 point_ps = _mm_cvtpd_ps(point);
	const __m128 px = _mm_set1_ps(point_ps[0]);
	const __m128 py = _mm_set1_ps(point_ps[1]);
	const __m128 range2 = _mm_set1_ps(float(range*range));
	for(const xy_interleave &curr : outline_simd)
		for(unsigned j = 0; j < curr.vecSize; j+=4)
		{
			__m128 dx = _mm_load_ps(curr.x + j) - px;
			__m128 dy = _mm_load_ps(curr.y + j) - py;
			__m128 cmp = _mm_cmplt_ps(dx*dx - range2, dy*dy);  // transform the inequality for more ILP
			if(_mm_movemask_ps(cmp))
				return true;
		}
#else
	// For efficiency, compare to range^2 instead of range.
	float range2 = range * range;
	float px = point.X();
	float py = point.Y();
	for(const xy_interleave &curr : outline_simd)
		for (unsigned j = 0; j < curr.vecSize; ++j)
		{
			float dx = curr.x[j] - px;
			float dy = curr.y[j] - py;
			if(dx*dx + dy*dy < range2)
				return true;
		}
#endif

	// FIXME: why did the original ignore the case of a tiny range
	// for a point near the centre of the mask, farther from any points?
//	if(range < radius)
//		return Contains(point);

	return false;
}



// Find out how close the given point is to the mask.
double Mask::Range(Point point, Angle facing) const
{
	// Rotate into the mask's frame of reference.
	point = (-facing).Rotate(point);
	if(Contains(point))
		return 0.;

#ifdef __SSE2__
	__m128 point_ps = _mm_cvtpd_ps(point);
	const __m128 px = _mm_set1_ps(point_ps[0]);
	const __m128 py = _mm_set1_ps(point_ps[1]);
	__m128 range2 = _mm_set1_ps(numeric_limits<float>::infinity());
	for(const xy_interleave &curr : outline_simd)
		for(unsigned j = 0; j < curr.vecSize; j+=4)
		{
			__m128 dx = _mm_load_ps(curr.x + j) - px;
			__m128 dy = _mm_load_ps(curr.y + j) - py;
			__m128 d2 = dx*dx + dy*dy;
			range2 = _mm_min_ps(range2, d2);
		}
	__m128 high64 = _mm_movehl_ps(px, range2);   // px is just a convenient dead variable to avoid a MOVAPS
	__m128 min2  = _mm_min_ps(range2, high64);   // last 2 elements
	__m128 sqrt2 = _mm_sqrt_ps(min2);            // force use of sqrtps instead of rsqrt + newton, for smaller code-size in this infrequently-used function
	__m128d double2 = _mm_cvtps_pd(sqrt2);
	return min(double2[0], double2[1]);
#else
	float range2 = numeric_limits<float>::infinity();
	float px = point.X();
	float py = point.Y();
	for(const xy_interleave &curr : outline_simd)
		for (unsigned j = 0; j < curr.vecSize; ++j)
		{
			float dx = curr.x[j] - px;
			float dy = curr.y[j] - py;
			range2 = min(range2, dx*dx + dy*dy);
		}
	return sqrtf(range2);
#endif
//	for(const Point &p : outline)
//		range2 = min(range2, p.DistanceSquared(point));

}


//#define INTERSECT_NONSIMD_OUTLINE
#ifdef INTERSECT_NONSIMD_OUTLINE
double Mask::Intersection(Point sA, Point vA) const
{
	// Keep track of the closest intersection point found.
	double closest = 1.;
	
	Point prev = outline.back();
	for(const Point &next : outline)
	{
		// Check if there is an intersection. (If not, the cross would be 0.) If
		// there is, handle it only if it is a point where the segment is
		// entering the polygon rather than exiting it (i.e. cross > 0).
		Point vB = next - prev;
		double cross = vB.Cross(vA);
		if(cross > 0.)
		{
			Point vS = prev - sA;
			double uB = vA.Cross(vS);
			double uA = vB.Cross(vS);
			// If the intersection occurs somewhere within this segment of the
			// outline, find out how far along the query vector it occurs and
			// remember it if it is the closest so far.
			if((uB >= 0.) & (uB < cross) & (uA >= 0.))
				closest = min(closest, uA / cross);
		}
		
		prev = next;
	}
	return closest;
}

#else // outline_simd

namespace {
	inline float __attribute__((unused)) Crossf(float Ax, float Ay, float Bx, float By)
	{
		return Ax * By - Ay * Bx;
	}

#ifdef __SSE2__
	inline __m128 Cross_ps(__m128 Ax, __m128 Ay, __m128 Bx, __m128 By)
	{
		return Ax * By - Ay * Bx;    // GNU vector extensions for operators instead of _mm functions
	}

	// result = newVal for elements where updateMask has the sign bit
	inline __m128 __attribute__((unused)) blendOnSignBit(__m128 old, __m128 newVal, __m128 updateMask)
	{
#ifdef __SSE41__
		return _mm_blendv_ps(old, newVal, updateMask);
#else
		__m128i iold = _mm_castps_si128(old);
		__m128i inew = _mm_castps_si128(newVal);
		__m128i iupdate = _mm_srai_epi32(_mm_castps_si128(updateMask), 31);  // broadcast the sign bit
		__m128i iblended = _mm_or_si128(
		    _mm_andnot_si128(iupdate, iold),
		    _mm_and_si128(iupdate, inew));
		return _mm_castsi128_ps(iblended);
#endif // SSE4.1
	}
#endif
}

double Mask::Intersection(Point sA, Point vA) const
{
#ifdef __SSE2__
	// Keep track of the closest intersection point found.
	__m128 closest = _mm_set1_ps(1.);

//	float sAx = sA.X(), sAy = sA.Y();
//	float vAx = vA.X(), vAy = vA.Y();
	__m128 sA_ps = _mm_cvtpd_ps(sA);
	const __m128 sAx_bcast = _mm_set1_ps(sA_ps[0]);
	const __m128 sAy_bcast = _mm_set1_ps(sA_ps[1]);
//	const __m128 vAx_bcast = _mm_set1_ps(vA.X());  // why are compilers so bad at this? 2x scalar converts.
//	const __m128 vAy_bcast = _mm_set1_ps(vA.Y());
	__m128 vA_ps = _mm_cvtpd_ps(vA);
	const __m128 vAx_bcast = _mm_set1_ps(vA_ps[0]);
	const __m128 vAy_bcast = _mm_set1_ps(vA_ps[1]);

	//float prev = outline.back();
	//float prevx = outline_simd.back().x[xy_interleave::vecSize-1];
	for(const xy_interleave &curr : outline_simd)
	{
		// cross == 0. means parallel.  cross > 0 means it's a point where the segment is
		// entering the polygon rather than exiting it; we only handle that case.
		for(unsigned i=0; i<xy_interleave::vecSize; ) {
			__m128 vBx = _mm_load_ps(curr.dx + i);  // (next-curr).X
			__m128 vBy = _mm_load_ps(curr.dy + i);

			__m128 cross = Cross_ps(vBx,vBy, vAx_bcast,vAy_bcast); //vB.Cross(vA);
			//__m128 crossPositive = _mm_cmpgt_ps(cross, _mm_setzero_ps());
			// early out if all elements have negative cross products
			// it's ok to do extra work in the rare case where an element has cross = +0.
			if(0b1111 != _mm_movemask_ps(cross))
			{
				__m128 vSx = _mm_load_ps(curr.x + i) - sAx_bcast; // Point vS = curr - sA;
				__m128 vSy = _mm_load_ps(curr.y + i) - sAy_bcast;
				__m128 uB = Cross_ps(vAx_bcast,vAy_bcast, vSx,vSy); // double uB = vA.Cross(vS);
				__m128 uA = Cross_ps(vBx,vBy, vSx,vSy);             // double uA = vB.Cross(vS);
				// If the intersection occurs somewhere within this segment of the
				// outline, find out how far along the query vector it occurs and
				// remember it if it is the closest so far.

				// TODO: camelCase these varnames somehow, even though they already include caps?
				__m128 uB_lt_cross = _mm_cmplt_ps(uB, cross);
				// use int vectors because we eventually feed into a shift to broadcast the sign bit
				__m128i uB_or_cross_neg = _mm_or_si128(_mm_castps_si128(uB), _mm_castps_si128(cross));
				__m128i uA_uB_or_cross_neg = _mm_or_si128(uB_or_cross_neg, _mm_castps_si128(uA));
				// only the sign bit is the truth value

				__m128i updateMin = _mm_andnot_si128(uA_uB_or_cross_neg, _mm_castps_si128(uB_lt_cross));
				// uB>=0. is true iff the sign bit is 0, except for negative 0.  We treat that as meaning less than 0 with underflow
				// (uB >= 0.) & (uB < cross) & (uA >= 0.) & (cross > 0)
				if(_mm_movemask_ps(_mm_castsi128_ps(updateMin)))
				{
					// divide + blend (without SSE4) are expensive enough to branch for,
					// and this is probably very rare

					// broadcast the sign bit to make elements of all-zero or all-one
					//__m128 mask = _mm_castsi128_ps(_mm_srai_epi32(updateMin, 31));

					// instead of blending after min, produce +Infinity in elements we don't want to update
					// SSE math doesn't slow down on NaN or Inf, so this is fine
					//cross = _mm_and_ps(mask, cross);
					// force uA to be positive, because -x / +0.0 is -Infinity
					//uA    = _mm_and_ps(uA, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)) );
					__m128 uA_over_cross = _mm_div_ps(uA, cross);
					__m128 newMin = _mm_min_ps(uA_over_cross, closest); // if unordered, min takes the 2nd operand
					closest = blendOnSignBit(closest, newMin, _mm_castsi128_ps(updateMin));
					if(_mm_movemask_ps(closest)) {
					    cerr << "badness in Interesct\n";
					    break;
					}
				}
			}
			i+=4;
		}
	}
	__m128 high64 = _mm_movehl_ps(vAx_bcast, closest);   // vAx is just a convenient dead variable to avoid a MOVAPS
	__m128 min64  = _mm_min_ps(closest, high64);
	return min(min64[0], min64[1]);

#else  // scalar version, might auto-vectorize with -ffast-math

	// Keep track of the closest intersection point found.
	float closest = 1.;

	float sAx = sA.X(), sAy = sA.Y();
	float vAx = vA.X(), vAy = vA.Y();
	//float prev = outline.back();
	//float prevx = outline_simd.back().x[xy_interleave::vecSize-1];
	for(const xy_interleave &curr : outline_simd)
	{
		// cross == 0. means parallel.  cross > 0 means it's a point where the segment is
		// entering the polygon rather than exiting it; we only handle that case.
		for(unsigned i=0; i<xy_interleave::vecSize; ) {
			//Point vB = next - prev;
			float vBx = curr.dx[i];  // **Already using dx is diff from this to next, not from prev to this
			float vBy = curr.dy[i];

			float cross = Crossf(vBx, vBy, vAx, vAy); //vB.Cross(vA);
			if(cross > 0.)
			{
				float vSx = curr.x[i] - sAx; // Point vS = curr - sA;
				float vSy = curr.y[i] - sAy;
				float uB = Crossf(vAx,vAy, vSx,vSy); // double uB = vA.Cross(vS);
				float uA = Crossf(vBx,vBy, vSx,vSy); // double uA = vB.Cross(vS);
				// If the intersection occurs somewhere within this segment of the
				// outline, find out how far along the query vector it occurs and
				// remember it if it is the closest so far.
				if((uB >= 0.) & (uB < cross) & (uA >= 0.))
					closest = min(closest, uA / cross);
			}
			++i;
		}
	}
	return closest;
#endif
}

#endif  // outline_simd for Intersection




// outline_simd version
bool Mask::Contains(Point point) const
{
	// If this point is contained within the mask, a ray drawn out from it will
	// intersect the mask an even number of times. If that ray coincides with an
	// edge, ignore that edge, and count all segments as closed at the start and
	// open at the end to avoid double-counting.

	// For simplicity, use a ray pointing straight downwards. A segment then
	// intersects only if its x coordinates span the point's coordinates.
#ifdef __SSE2__
	__m128 point_ps = _mm_cvtpd_ps(point);
	const __m128 Vx = _mm_set1_ps(point_ps[0]);
	const __m128 Vy = _mm_set1_ps(point_ps[1]);

	// Instead of actually adding and checking odd/even, just XOR (add-without-carry) in the sign bit of each element
	__m128 intersections = _mm_setzero_ps();

	for(const xy_interleave &curr : outline_simd)
	{
		for(unsigned i=0; i<xy_interleave::vecSize; )
		{
			__m128 currx = _mm_load_ps(curr.x + i);
			__m128 segx  = _mm_load_ps(curr.dx + i);
			__m128 nextx = currx + segx;

			__m128 cmpCurr = _mm_cmplt_ps(Vx, currx);
			__m128 cmpNext = _mm_cmplt_ps(Vx, nextx);
			__m128 xIntersect = _mm_xor_ps(cmpCurr, cmpNext); // gte one, less than the other
			if(true || _mm_movemask_ps(xIntersect))  // early out check if all points fail
			{
				__m128 curry = _mm_load_ps(curr.y + i);
				__m128 segy = _mm_load_ps(curr.dy + i);

				__m128 dcx = Vx - currx;
				__m128 dcy = Vy - curry;
				__m128 yGEpoint = _mm_cmpge_ps(segy * dcx, segx * dcy);
				yGEpoint = _mm_xor_ps(yGEpoint, currx); // the inequality is inverted if curr.x is negative
				yGEpoint = _mm_and_ps(yGEpoint, xIntersect);
				// sign bits have the truth values, we can use it directly without a compare
				intersections = _mm_xor_ps(intersections, yGEpoint);
			}
			i+=4;
		}
	}
	unsigned mask = _mm_movemask_ps(intersections);
	mask ^= mask >> 2;   // parity of the 4-bit mask.  x86 has a parity flag, but IDK how to get a compiler to use it
	mask ^= mask >> 1;
	return mask & 1;
#else
	bool intersections_odd = false;
	const float Vx = point.X();
	const float Vy = point.Y();
	for(const xy_interleave &curr : outline_simd)
	{
		for(unsigned i=0; i<xy_interleave::vecSize; )
		{
			float nextX = curr.x[i] + curr.dx[i];
			if ((curr.x[i] <= Vx) == (Vx < nextX))
			{
				float dcx = Vx - curr.x[i];
				float dcy = Vy - curr.y[i];
			// Avoid a division by multiplying both sides of the inequality, and bring both seg.X terms to one side
			// if fast-math disables denormals, seg.X can be 0 with dp.X not quite zero
			// This leads to a miscount, but no crash.
			// It can happen if the point is at the same x as n and p, but with a different y
			// however, it still only happens if the point is inside the mask radius

			// Our trace algo steps by 0.25, so this is not a concern in practice
				//bool yGEpoint = seg.Y() * dp.X() >= dp.Y() * seg.X();
				bool yGEpoint = (curr.dy[i] * dcx >= curr.dx[i] * dcy);
				yGEpoint ^= signbit(curr.dx[i]);
				// Multiplying both sides inverted the inequality if seg.X() is negative
				intersections_odd ^= yGEpoint;
			}
			i++;
		}
	}
	// If the number of intersections is odd, the point is within the mask.
	return intersections_odd;
#endif
}

/*
  seg = next - prev;
  dn = V - next;

  dp = dn_last_iter;		// V - prev
  prev = next_last_iter;

  signbit(dp) == 0;	  // prev.X <= point.X;     // 0 <= point.X - prev.X
  signbit(dn) == 1;	  // point.X < next.X       // point.X - next.X < 0

  // (p.X <= V.X) == (V.X < c.X)
  signbit(dp) ^ signbit(dn) == 1;


  y = prev.Y + seg.Y * dp.X / seg.X
  intersections += prev.Y + seg.Y * dp.X / seg.X >= V.Y

  if(signbit(dp) ^ signbit(dn) == 1)
    intersections += (prev.Y * seg.X + seg.Y * dp.X >= V.Y * seg.X) ^ signbit(seg.X)

  if(signbit(dp) != signbit(dn)) {
    //intersections += (seg.Y * dp.X >= V.Y * seg.X - prev.Y * seg.X) ^ signbit(seg.X)
    //intersections += (seg.Y * dp.X >= (V.Y - prev.Y) * seg.X) ^ signbit(seg.X)
      intersections += (seg.Y * dp.X >= dp.Y * seg.X) ^ signbit(seg.X)
  }
 */


#if 1 //def BENCHMARK_MASK

/*
class TestMask: public Mask {
public:
	double Intersection(Point sA, Point vA) const = default;
	bool Contains(Point point) const = default;
public:
	size_t OutlineCount() const { return outline.size(); }
	double GetRadius() const { return radius; }
	const std::vector<Point> &Outline() const { return outline; }
};
*/

#include <stdio.h>
#include <float.h>
//#include "Random.h"

extern "C" int test_main(int argc, char*argv[])
{
	Mask msk;
	if (argc <= 1){
		ImageBuffer *kes_img = ImageBuffer::Read("images/ship/kestrel.png");
		if (!kes_img) {
			perror("reading kestrel image");
			return 1;
		}
		msk.Create(kes_img);
	}

	while (--argc) {
		ImageBuffer *img = ImageBuffer::Read(argv[argc]);
		if (!img) {
			perror("reading image from ship");
			return 1;
		}
		msk.Create(img);
		delete img;
		printf("%s : %lu\n", argv[argc], msk.OutlineCount());

		auto it = msk.Outline().cbegin();
		for(auto& curr : msk.Outline_simd())
		{
			// std::numeric_limits<double>::min()
//			if (fabs( (prev-curr).X() ) <= 100*DBL_MIN && prev.X() != curr.X()) {
			for(unsigned j=0; j<curr.vecSize; ++j) {
				printf("   %8g\t%8g\t\tdx=%g\tdy=%g\n", curr.x[j], curr.y[j], curr.dx[j], curr.dy[j]);
				if(it !=  msk.Outline().cend()) {
					printf("   %8g\t%8g\n", it->X(), it->Y());
					++it;
				}
			}
		}

	}


	vector<Point> points;
	double mskradius = msk.GetRadius();
	for (int i=0 ; i<8 ; ++i) {
//		points.emplace_back(Random::Real() * 2*mskradius, Random::Real() * 2*mskradius);
		points.emplace_back(i * 0.125 * mskradius,  i * 0.125 * mskradius);
	}
	int total = 0;
	for (int i=0; i<1000000; i++) {
		for (Point &p : points)
			total += msk.Contains(p);
	}
	printf("total contain hits = %d\n", total);
	return 0;
}

# define weak_alias(name, aliasname) _weak_alias (name, aliasname)
# define _weak_alias(name, aliasname) \
  extern __typeof (name) aliasname __attribute__ ((weak, alias (#name)));

weak_alias(test_main, main)
#endif
