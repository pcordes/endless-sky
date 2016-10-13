/* Point.h
Copyright (c) 2014 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#ifndef POINT_H_
#define POINT_H_

#ifdef __SSE2__
#include <pmmintrin.h>       // We use some SSE3 intrinsics if it's also enabled.
#define POD_POINT            // For auto-vectorization, non-POD may be better, passing by ref
#endif

// Class representing a 2D point with functions for a variety of vector operations.
// A Point can represent either a location or a vector (e.g. a velocity, or a
// distance between two points, or a unit vector representing a direction). All
// basic mathematical operations that make sense for vectors are supported.
// Internally the coordinates are stored in a SSE vector and the processor's vector
// extensions are used to optimize all operations.
class Point {
public:
	Point();
	Point(double x, double y);
#ifndef POD_POINT
	Point(const Point &point);  // no need to make this a non-POD type
	// non-POD may actually improve pass & return by-value.
	// With the union definition, x and y go in separate registers
	//Point &operator=(const Point &point);   // why overload this at all?
#endif

	// Check if the point is anything but (0, 0).
	explicit operator bool() const;
	bool operator!() const;
	
	// No comparison operators are provided because I never expect to use them
	// and because comparisons with doubles are inherently unsafe due to the
	// possibility of rounding errors and imprecision.
	
	Point operator+(const Point &point) const;
	Point &operator+=(const Point &point);
	Point operator-(const Point &point) const;
	Point &operator-=(const Point &point);
	Point operator-() const;
	
	Point operator*(double scalar) const;
	friend Point operator*(double scalar, const Point &point);
	Point &operator*=(double scalar);
	Point operator/(double scalar) const;
	Point &operator/=(double scalar);
	
	Point operator*(const Point &other) const;
	Point &operator*=(const Point &other);
	
	double &X();
	double X() const;
	double &Y();
	double Y() const;
	
	void Set(double x, double y);
	
	// Operations that treat this point as a vector from (0, 0):
	double Dot(const Point &point) const;
	double Cross(const Point point) const;
	
	double Length() const;
	double LengthSquared() const;
	Point Unit() const;
	bool InRange(double range) const { return LengthSquared() <= (range*range); }  // some use cases are <=, some are <
	bool OutOfRange(double range) const { return LengthSquared() > (range*range); }
	
	double Distance(const Point &point) const;
	double DistanceSquared(const Point &point) const;
	bool InRange(const Point &other, double range) const { return DistanceSquared(other) <= (range*range); }
	bool OutOfRange(const Point &other, double range) const { return DistanceSquared(other) > (range*range); }
	
	friend Point abs(const Point &p);
	friend Point min(const Point &p, const Point &q);
	friend Point max(const Point &p, const Point &q);
	
	double &access(int idx);
	
private:

#ifdef __SSE2__
public:
	// Private constructor, using a vector.
	Point(const __m128d &v);
	operator __m128d() const { return v; }

private:
#if 0
	union {
		__m128d v;
		struct {
			double x;
			double y;
		};
		//double xy[2];
		//double &Point::access(int idx)       { return xy[idx]; }
	};
#else
	__m128d v;
#endif
#else
	alignas(16) double x;  // should help with auto-vectorization for non-x86 targets
	double y;
#endif
};



// Inline accessor functions, for speed:
#ifdef __SSE2__
// avoid using the x and y members at all, so we don't need a union
// this lets Point be passed/returned by value in an XMM register (if we make it POD),
// rather than in two separate XMM registers for scalar x and y

// http://stackoverflow.com/questions/26554829/how-to-access-simd-vector-elements-when-overloading-array-access-operators
inline double &Point::X()       { return reinterpret_cast<double &>(v); }
inline double  Point::X() const { return v[0]; }
inline double &Point::Y()       { return *(reinterpret_cast<double *>(&v) + 1); }  // v[1] doesn't work
inline double  Point::Y() const { return v[1]; }
#else
inline double &Point::X()       { return x; }
inline double  Point::X() const { return x; }
inline double &Point::Y()       { return y; }
inline double  Point::Y() const { return y; }
#endif


#ifndef __SSE2__
#include <algorithm>
#endif

#include <cmath>


inline Point::Point()
#ifdef __SSE2__
	: v(_mm_setzero_pd())
#else
	: x(0.), y(0.)
#endif
{
}



inline Point::Point(double x, double y)
#ifdef __SSE2__
	: v(_mm_setr_pd(x, y))
#else
	: x(x), y(y)
#endif
{
}


#ifndef POD_POINT
inline Point::Point(const Point &point)
#ifdef __SSE2__
	: v(point.v)
#else
	: x(point.x), y(point.y)  // TODO: report gcc perf bug about scalar store/reloads with this in SSE2 m
#endif
{
}
#endif



// Check if the point is anything but (0, 0).
inline Point::operator bool() const
{
	return !!*this;
}

inline bool Point::operator!() const
{
	// TODO: SIMD?
	return (!X() & !Y());
}



inline Point Point::operator+(const Point &point) const
{
	Point result = *this;
	return result += point;
}
inline Point &Point::operator+=(const Point &point)
{
#ifdef __SSE2__
	v += point.v;
#else
	x += point.x;
	y += point.y;
#endif
	return *this;
}



inline Point Point::operator-(const Point &point) const
{
	Point result = *this;
	return result -= point;
}
inline Point &Point::operator-=(const Point &point)
{
#ifdef __SSE2__
	v -= point.v;
#else
	x -= point.x;
	y -= point.y;
#endif
	return *this;
}

inline Point Point::operator-() const   // unary
{
	return Point() - *this;
}



inline Point Point::operator*(double scalar) const
{
	Point result = *this;
	return result *= scalar;
}
inline Point operator*(double scalar, const Point &point)
{
	return point * scalar;
}
inline Point &Point::operator*=(double scalar)
{
#ifdef __SSE2__
	v *= _mm_set1_pd(scalar);
#else
	x *= scalar;
	y *= scalar;
#endif
	return *this;
}



inline Point Point::operator*(const Point &other) const
{
	Point result = *this;
	return result *= other;
}
inline Point &Point::operator*=(const Point &other)
{
#ifdef __SSE2__
	v *= other.v;
#else
	x *= other.x;
	y *= other.y;
#endif
	return *this;
}



inline Point Point::operator/(double scalar) const
{
	Point result = *this;
	return result /= scalar;
}
inline Point &Point::operator/=(double scalar)
{
#ifdef __SSE2__
	v /= _mm_set1_pd(scalar);
#else
	x /= scalar;
	y /= scalar;
#endif
	return *this;
}



inline void Point::Set(double x, double y)
{
#ifdef __SSE2__
	v = _mm_set_pd(y, x);
#else
	this->x = x;
	this->y = y;
#endif
}


#ifdef __SSE2__
static inline
double hsum_pd(__m128d vec)
{
#if 0 && defined(__SSE3__) && !defined(__AVX__)
	// HADDPD is only possibly worth it when it also saves a MOVAPD (i.e. without AVX)
	vec = _mm_hadd_pd(vec, vec);
	return _mm_cvtsd_f64(vec);
	//#error xd
#else
//	return vec[0] + vec[1];   // Let the compiler choose, using GNU C vector extensions syntax
//	__m128d swapped = _mm_shuffle_pd(vec, vec, 0x01);

	__m128d high = _mm_unpackhi_pd(vec, vec);
	return _mm_cvtsd_f64(vec) + _mm_cvtsd_f64(high);
#endif
}
#endif

// Operations that treat this point as a vector from (0, 0):
inline double Point::Dot(const Point &point) const
{
#ifdef __SSE2__
	__m128d prod = v * point.v;
#if 0
	return hsum_pd(prod);
#elif 0
	return prod[0] + prod[1];
#else
	// MOVHLPS into a dead register is the only way to hsum without an extra
	// MOVAPD, when AVX is unavailable.
	// using a copy of v as our dead register gives good results sometimes.
	__m128 tmp = _mm_castpd_ps(v);
	__m128d high = _mm_castps_pd(_mm_movehl_ps(tmp, _mm_castpd_ps(prod)));
	return _mm_cvtsd_f64(prod) + _mm_cvtsd_f64(high);
	// TODO: something that leaves the sum in both halves?
	// so smart compilers can avoid re-broadcasting, like Unit() does
#endif

#else
	return x * point.x + y * point.y;
#endif
}



inline double Point::Cross(const Point other) const
{
#ifdef __SSE2__
	__m128d otherSwapped = _mm_shuffle_pd(other.v, other.v, 0x01);
	__m128d crossmul = otherSwapped * v;

#if 0 && defined(__SSE3__) && !defined(__AVX__)
	// never worth using
	__m128d hsum = _mm_hsub_pd(crossmul, crossmul);
	return _mm_cvtsd_f64(hsum);
#else
//	return crossmul[0] - crossmul[1];  // letting the compiler see this scalar sub enables some optimizations (like compare to zero by actually comparing vs. each other)
	__m128 tmp = _mm_castpd_ps(v);  // avoid a movapd when AVX is unavailable
	__m128d high = _mm_castps_pd(_mm_movehl_ps(tmp, _mm_castpd_ps(crossmul)));
	return _mm_cvtsd_f64(crossmul) - _mm_cvtsd_f64(high);
	// TODO: don't use tmp when AVX is available: it may hurt slightly, here and in Dot

//	__m128d swapped = _mm_shuffle_pd(crossmul, crossmul, 0x01);
//	crossmul -= swapped;
//	return _mm_cvtsd_f64(crossmul);
#endif

#else
	return x * other.y - y * other.x;
#endif
}



inline double Point::Length() const
{
	// maybe do this in a way that allows CSE in functions that do velocity.Length() and .Unit().
	return sqrt(LengthSquared());  // without -ffast-math, using _mm_sqrt_pd is better
}



inline double Point::LengthSquared() const
{
	return Dot(*this);
}



inline Point Point::Unit() const
{
#ifdef __SSE2__
	__m128d square = v * v;
#if defined(__SSE3__) && !defined(__AVX__)
	__m128d hsum = _mm_hadd_pd(square, square);
#else
	__m128d swapped = _mm_shuffle_pd(square, square, 0x01);
	__m128d hsum     = square + swapped;
#endif
	// There's no double-precision equivalent of rsqrtps
	__m128d length = _mm_sqrt_pd(hsum);
	return Point(v / length);
#else  // scalar
	double b = 1. / sqrt(x * x + y * y);
	return Point(x * b, y * b);
#endif
}



inline double Point::Distance(const Point &point) const
{
	return (*this - point).Length();
}

inline double Point::DistanceSquared(const Point &point) const
{
	return (*this - point).LengthSquared();
}



// Absolute value of both coordinates.
inline Point abs(const Point &p)
{
#ifdef __SSE2__
	// Absolute value for doubles just involves clearing the sign bit.
	const __m128d sign_mask = _mm_set1_pd(-0.);
	return Point(_mm_andnot_pd(sign_mask, p.v));
#else
	return Point(std::abs(p.x), std::abs(p.y));
#endif
}



// Take the min of the x and y coordinates.
inline Point min(const Point &p, const Point &q)
{
#ifdef __SSE2__
	return Point(_mm_min_pd(p.v, q.v));
#else
	return Point(std::min(p.x, q.x), std::min(p.y, q.y));
#endif
}



// Take the max of the x and y coordinates.
inline Point max(const Point &p, const Point &q)
{
#ifdef __SSE2__
	return Point(_mm_max_pd(p.v, q.v));
#else
	return Point(std::max(p.x, q.x), std::max(p.y, q.y));
#endif
}



#ifdef __SSE2__
// Private constructor, using a vector.
inline Point::Point(const __m128d &v)
	: v(v)
{
}
#endif

#endif  // POINT_H_

