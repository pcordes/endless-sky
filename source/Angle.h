/* Angle.h
Copyright (c) 2014 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#ifndef ANGLE_H_
#define ANGLE_H_

#include "Point.h"

#include <cstdint>
#include "pi.h"

// Represents an angle, in degrees. Angles are in "clock" orientation rather
// than usual mathematical orientation. That is, 0 degrees is up, and angles
// increase in a clockwise direction. Angles can be efficiently mapped to unit
// vectors, which also makes rotating a vector an efficient operation.
class Angle {
public:
	// Return a random angle up to the given amount (between 0 and 360).
	static Angle Random();
	static Angle Random(double range);
	
	
public:
	// The default constructor creates an angle pointing up (zero degrees).
	Angle();
	// Construct an Angle from the given angle in degrees. Allow this conversion
	// to be implicit to allow syntax like "angle += 30".
	Angle(double degrees);
	// Construct an angle pointing in the direction of the given vector.
	explicit Angle(const Point &point);
	
	// Mathematical operators.
	Angle operator+(const Angle &other) const;
	Angle &operator+=(const Angle &other);
	Angle operator-(const Angle &other) const;
	Angle &operator-=(const Angle &other);
	Angle operator-() const;
	
	// Get a unit vector in the direction of this angle.
	const Point &Unit() const;
	// Convert an Angle object to degrees, in the range -180 to 180.
	double Degrees() const;
	
	// Return a point rotated by this angle around (0, 0).
	Point Rotate(const Point &point) const;
	
private:
	explicit Angle(int32_t angle);
	
	
private:
	// The angle is stored as an integer value between 0 and 2^16 - 1. This is
	// so that any angle can be mapped to a unit vector (a very common operation)
	// with just a single array lookup. It also means that "wrapping" angles
	// to the range of 0 to 360 degrees can be done via a bit mask.
	int32_t angle;

	static const int32_t STEPS = 0x10000;
	static const int32_t MASK = STEPS - 1;
	static constexpr double DEG_TO_STEP = STEPS / 360.;
	static constexpr double STEP_TO_RAD = PI / (STEPS / 2);

	class UnitVectorCache {
	public:  // this is still part of the private implementation of Angle
		UnitVectorCache();          // construct the sincos lookup table
		alignas(64) Point lut[STEPS];
	};
        static const UnitVectorCache unitVectorCache;
};

// Default constructor: generates an angle pointing straight up.
inline Angle::Angle()
	: angle(0)
{
}

// Convert an angle in degrees into an Angle object.
inline Angle::Angle(double degrees)
	: angle(static_cast<int64_t>(degrees * DEG_TO_STEP + .5) & MASK)
{
}

// Construct an angle pointing in the direction of the given vector.
inline Angle::Angle(const Point &point)
	: Angle(TO_DEG * atan2(point.X(), -point.Y()))
{
}

inline Angle Angle::operator+(const Angle &other) const
{
	Angle result = *this;
	result += other;
	return result;
}

inline Angle &Angle::operator+=(const Angle &other)
{
	angle += other.angle;
	angle &= MASK;
	return *this;
}

inline Angle Angle::operator-(const Angle &other) const
{
	Angle result = *this;
	result -= other;
	return result;
}

inline Angle &Angle::operator-=(const Angle &other)
{
	angle -= other.angle;
	angle &= MASK;
	return *this;
}

inline Angle Angle::operator-() const
{
	return Angle((-angle) & MASK);
}


// we could cut the table size in half with float
// and possibly in half again by calculating cos() as sqrt(1 - sin^2), and setting the sign based on the angle quadrant
inline const Point &Angle::Unit() const
{
	return unitVectorCache.lut[angle];
}


// Return a point rotated by this angle around (0, 0).
// called frequently, and not as complicated as it looks
inline Point Angle::Rotate(const Point &point) const
{
	Point unit = Unit();
#ifdef __SSE2__
				//vector halves(shuffle index)    [low(0) | high(1) ]
	__m128d unitNegX = _mm_xor_pd(unit, _mm_setr_pd(-0., 0.));       //-ux| uy
	__m128d swappedUnitNegX = _mm_shuffle_pd(unitNegX, unitNegX, 1); // uy|-ux
	// rotating different points by the same angle is sometimes done in a loop, where swappedUnitNegX can CSE

	__m128d cross = _mm_mul_pd(swappedUnitNegX, point);    // [ uy*px   | -ux*py ]
	__m128d vert  = _mm_mul_pd(unit, point);               // [ ux*px   |  uy*py ]

	__m128d merge1 = _mm_move_sd(vert, cross);	       // [ uy*px   |  uy*py ]  // shuffle(c, v, 0b10)
	__m128d merge2 = _mm_shuffle_pd(cross, vert, 0b01);    // [-ux*py   |  ux*px ]

	// SSE3 _mm_addsub_pd would still need an XOR (to negate both elements of one vector)
	__m128d result = _mm_sub_pd(merge2, merge1);
	return Point(result);                   	  // [ -UxPy - UyPx | UxPx - UyPy ]
#else
	// If using the normal mathematical coordinate system, this would be easier.
	// Since we're not, the math is a tiny bit less elegant:
	return Point(-unit.X() * point.Y() - unit.Y() * point.X(),
		      unit.X() * point.X() - unit.Y() * point.Y());
#endif
}

// Constructor using Angle's internal representation.
inline Angle::Angle(int32_t angle)
	: angle(angle)
{
}
#endif

/* rotate: clang output
        movapd  xmm1, xmmword ptr [rax + _ZN5Angle15unitVectorCacheE]
        movapd  xmm2, xmmword ptr [rsi]

1c int? movapd  xmm0, xmm1 	// 1c integer domain on Merom??
5c      mulpd   xmm0, xmm2	// 5c
# vertprod ready in 6c  (or 5c with zero-latency mov)

1c      shufpd  xmm2, xmm2, 1           # xmm2 = xmm2[1,0]
5c      mulpd   xmm2, xmm1
cross in 6c (or 7c from resource conflict)

1c      movapd  xmm1, xmm2
1c      shufpd  xmm1, xmm0, 1           # xmm1 = xmm1[1],xmm0[0]
merge in 8c (or 9c RC)

1c int  xorpd   xmm2, xmmword ptr [rip + .LCPI0_0]   # Integer domain on Core2
negcross in bypass delay + 7c (or 8c RC)
1c int  movsd   xmm0, xmm2              # any port
merge2   in 8c (or 9c RC)

bypass delay on input from movsd?
3c      subpd   xmm0, xmm1
result ready in 11c (or 12c RC)

 */
