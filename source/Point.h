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

#ifdef __SSE3__
#include <pmmintrin.h>
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
	//Point(const Point &point);  // no need to make this a non-POD type
	Point(const Point &point) : x(point.x), y(point.y) {}  // non-POD may actually improve return-by-value?

	//Point &operator=(const Point &point);   // why overload this at all?

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
	const double &X() const;
	double &Y();
	const double &Y() const;
	
	void Set(double x, double y);
	
	// Operations that treat this point as a vector from (0, 0):
	double Dot(const Point &point) const;
	double Cross(const Point &point) const;
	
	double Length() const;
	double LengthSquared() const;
	Point Unit() const;
	
	double Distance(const Point &point) const;
	double DistanceSquared(const Point &point) const;
	
	friend Point abs(const Point &p);
	friend Point min(const Point &p, const Point &q);
	friend Point max(const Point &p, const Point &q);
	
	
private:
#ifdef __SSE3__
	// Private constructor, using a vector.
	Point(const __m128d &v);
	
	
private:
	union {
		__m128d v;
		struct {
			double x;
			double y;
		};
	};
#else
    alignas(16)	double x;  // enable auto-vectorization for more stuff (even without SSE3)
	double y;
#endif
};



// Inline accessor functions, for speed:
inline       double &Point::X()       { return x; }
inline const double &Point::X() const { return x; }
inline       double &Point::Y()       { return y; }
inline const double &Point::Y() const { return y; }

#define INLINE_POINT
#ifdef INLINE_POINT

#ifndef __SSE3__
#include <algorithm>
#include <cmath>
using namespace std;
#endif



inline Point::Point()
#ifdef __SSE3__
	: v(_mm_setzero_pd())
#else
	: x(0.), y(0.)
#endif
{
}



inline Point::Point(double x, double y)
#ifdef __SSE3__
	: v(_mm_set_pd(y, x))
#else
	: x(x), y(y)
#endif
{
}


/*
inline Point::Point(const Point &point)
#ifdef __SSE3__
	: v(point.v)
#else
	: x(point.x), y(point.y)
#endif
{
}
*/


/*
inline Point &Point::operator=(const Point &point)
{
#ifdef __SSE3__
	v = point.v;
#else
	x = point.x;
	y = point.y;
#endif
	return *this;
}
*/


// Check if the point is anything but (0, 0).
inline Point::operator bool() const
{
	return !!*this;
}



inline bool Point::operator!() const
{
	return (!x & !y);
}



inline Point Point::operator+(const Point &point) const
{
#ifdef __SSE3__
	return Point(v + point.v);
#else
	return Point(x + point.x, y + point.y);
#endif
}



inline Point &Point::operator+=(const Point &point)
{
#ifdef __SSE3__
	v += point.v;
#else
	x += point.x;
	y += point.y;
#endif
	return *this;
}



inline Point Point::operator-(const Point &point) const
{
#ifdef __SSE3__
	return Point(v - point.v);
#else
	return Point(x - point.x, y - point.y);
#endif
}



inline Point &Point::operator-=(const Point &point)
{
#ifdef __SSE3__
	v -= point.v;
#else
	x -= point.x;
	y -= point.y;
#endif
	return *this;
}



inline Point Point::operator-() const
{
	return Point() - *this;
}



inline Point Point::operator*(double scalar) const
{
#ifdef __SSE3__
	return Point(v * _mm_loaddup_pd(&scalar));
#else
	return Point(x * scalar, y * scalar);
#endif
}



inline Point operator*(double scalar, const Point &point)
{
#ifdef __SSE3__
	return Point(point.v * _mm_loaddup_pd(&scalar));
#else
	return Point(point.x * scalar, point.y * scalar);
#endif
}



inline Point &Point::operator*=(double scalar)
{
#ifdef __SSE3__
	v *= _mm_loaddup_pd(&scalar);
#else
	x *= scalar;
	y *= scalar;
#endif
	return *this;
}



inline Point Point::operator*(const Point &other) const
{
#ifdef __SSE3__
	Point result;
	result.v = v * other.v;
	return result;
#else
	return Point(x * other.x, y * other.y);
#endif
}



inline Point &Point::operator*=(const Point &other)
{
#ifdef __SSE3__
	v *= other.v;
#else
	x *= other.x;
	y *= other.y;
#endif
	return *this;
}



inline Point Point::operator/(double scalar) const
{
#ifdef __SSE3__
	return Point(v / _mm_loaddup_pd(&scalar));
#else
	return Point(x / scalar, y / scalar);
#endif
}



inline Point &Point::operator/=(double scalar)
{
#ifdef __SSE3__
	v /= _mm_loaddup_pd(&scalar);
#else
	x /= scalar;
	y /= scalar;
#endif
	return *this;
}



inline void Point::Set(double x, double y)
{
#ifdef __SSE3__
	v = _mm_set_pd(y, x);
#else
	this->x = x;
	this->y = y;
#endif
}



// Operations that treat this point as a vector from (0, 0):
inline double Point::Dot(const Point &point) const
{
#ifdef __SSE3__
	__m128d b = v * point.v;
	b = _mm_hadd_pd(b, b);
	return reinterpret_cast<double &>(b);
#else
	return x * point.x + y * point.y;
#endif
}



inline double Point::Cross(const Point &point) const
{
#ifdef __SSE3__
	__m128d b = _mm_shuffle_pd(point.v, point.v, 0x01);
	b *= v;
	b = _mm_hsub_pd(b, b);
	return reinterpret_cast<double &>(b);
#else
	return x * point.y - y * point.x;
#endif
}



inline double Point::Length() const
{
#ifdef __SSE3__
	__m128d b = v * v;
	b = _mm_hadd_pd(b, b);
	b = _mm_sqrt_pd(b);
	return reinterpret_cast<double &>(b);
#else
	return sqrt(x * x + y * y);
#endif
}



inline double Point::LengthSquared() const
{
	return Dot(*this);
}



inline Point Point::Unit() const
{
#ifdef __SSE3__
	__m128d b = v * v;
	b = _mm_hadd_pd(b, b);
	b = _mm_sqrt_pd(b);
	return Point(v / b);
#else
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
#ifdef __SSE3__
	// Absolute value for doubles just involves clearing the sign bit.
	static const __m128d sign_mask = _mm_set1_pd(-0.);
    return Point(_mm_andnot_pd(sign_mask, p.v));
#else
	return Point(abs(p.x), abs(p.y));
#endif
}



// Take the min of the x and y coordinates.
inline Point min(const Point &p, const Point &q)
{
#ifdef __SSE3__
	return Point(_mm_min_pd(p.v, q.v));
#else
	return Point(min(p.x, q.x), min(p.y, q.y));
#endif
}



// Take the max of the x and y coordinates.
inline Point max(const Point &p, const Point &q)
{
#ifdef __SSE3__
	return Point(_mm_max_pd(p.v, q.v));
#else
	return Point(max(p.x, q.x), max(p.y, q.y));
#endif
}



#ifdef __SSE3__
// Private constructor, using a vector.
inline inline Point::Point(const __m128d &v)
	: v(v)
{
}
#endif

#endif // INLINE_POINT


#endif  // POINT_H_

