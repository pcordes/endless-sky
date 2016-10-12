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
	// If using the normal mathematical coordinate system, this would be easier.
	// Since we're not, the math is a tiny bit less elegant:
	Point unit = Unit();
	return Point(-unit.Y() * point.X() - unit.X() * point.Y(),
		-unit.Y() * point.Y() + unit.X() * point.X());
}


// Constructor using Angle's internal representation.
inline Angle::Angle(int32_t angle)
	: angle(angle)
{
}
#endif
