/* Angle.cpp
Copyright (c) 2014 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#include "Angle.h"

#include "pi.h"
#include "Random.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace std;

namespace {
	// Suppose you want to be able to turn 360 degrees in one second. Then you are
	// turning 6 degrees per time step. If the Angle lookup is 2^16 steps, then 6
	// degrees is 1092 steps, and your turn speed is accurate to +- 0.05%. That seems
	// plenty accurate to me. At that step size, the lookup table is exactly 1 MB.
	static const int32_t STEPS = 0x10000;
	static const int32_t MASK = STEPS - 1;
	static const double DEG_TO_STEP = STEPS / 360.;
	static const double STEP_TO_RAD = PI / (STEPS / 2);
}



// Get a random angle.
Angle Angle::Random()
{
	return Angle(static_cast<int32_t>(Random::Int(STEPS)));
}


// same behaviour if this is a nested class & member in Angle
class foo {
public:
	foo();
	alignas(64) float lut[STEPS];
};
static const foo test_primitive;

//Point Angle::unitVectorCache.lut[STEPS];  // only the instance is static
foo::foo() {
	// no wasted memset
	for(int i = 0; i < STEPS; ++i) {
		float radians = i * STEP_TO_RAD;
		lut[i] = sinf(radians);
	}
}



// Get a random angle between 0 and the given number of degrees.
Angle Angle::Random(double range)
{
	// The given range would have to be about 22.6 million degrees to overflow
	// the size of a 32-bit int, which should never happen in normal usage.
	uint32_t mod = static_cast<uint32_t>(fabs(range) * DEG_TO_STEP) + 1;
	return Angle(mod ? static_cast<int32_t>(Random::Int(mod)) : 0);
}


/*
const float Angle::testarray[14] = { 1, 2, 3, 4 };
const Point Angle::unitCache_flat[2] = { {1,2}, {2, 3} };

Point Angle::testUnit_byval() const
{
	return unitVectorCache.lut[angle];
}
const Point &Angle::testUnit_byref() const
{
	return unitVectorCache.lut[angle];
}
*/

Angle::UnitVectorCache::UnitVectorCache()
{
	// lut[] is zeroed before this loop runs, by the Point() default constructor :(
	for(int i = 0; i < STEPS; ++i)
	{
		double radians = i * STEP_TO_RAD;
		// The graphics use the usual screen coordinate system, meaning that
		// positive Y is down rather than up. Angles are clock angles, i.e.
		// 0 is 12:00 and angles increase in the clockwise direction. So, an
		// angle of 0 degrees is pointing in the direction (0, -1).
		lut[i] = { sin(radians), -cos(radians) };
//		cache[i] = Point(sin(radians), -cos(radians));
	}
}


/*
Point Angle::UnitVectorCache::lut[STEPS];  // static member of the nested class

Angle::UnitVectorCache::UnitVectorCache()
{
	for(int i = 0; i < STEPS; ++i)
	{
		double radians = i * STEP_TO_RAD;
		// The graphics use the usual screen coordinate system, meaning that
		// positive Y is down rather than up. Angles are clock angles, i.e.
		// 0 is 12:00 and angles increase in the clockwise direction. So, an
		// angle of 0 degrees is pointing in the direction (0, -1).
		lut[i] = { sin(radians), -cos(radians) };
//		cache[i] = Point(sin(radians), -cos(radians));
	}


}

Point Angle::testUnit() const
{
	return UnitVectorCache::lut[angle];
}
*/


/*
// Get a unit vector in the direction of this angle.
Point Angle::Unit() const
{
	// The very first time this is called, create a lookup table of unit vectors.
	static vector<Point> cache;
	if(cache.empty())
	{
		cache.reserve(STEPS);
		for(int i = 0; i < STEPS; ++i)
		{
			double radians = i * STEP_TO_RAD;
			// The graphics use the usual screen coordinate system, meaning that
			// positive Y is down rather than up. Angles are clock angles, i.e.
			// 0 is 12:00 and angles increase in the clockwise direction. So, an
			// angle of 0 degrees is pointing in the direction (0, -1).
			cache.emplace_back(sin(radians), -cos(radians));
		}
	}
	return cache[angle];
}
*/

/*
static Point unitVectors[STEPS];
static constexpr Point* BuildUnitCache()
{
	for(int i = 0; i < STEPS; ++i)
	{
		double radians = i * STEP_TO_RAD;
		// The graphics use the usual screen coordinate system, meaning that
		// positive Y is down rather than up. Angles are clock angles, i.e.
		// 0 is 12:00 and angles increase in the clockwise direction. So, an
		// angle of 0 degrees is pointing in the direction (0, -1).
//		unitVectors[i] = { sin(radians), -cos(radians) };
//		unitVectors[i] = Point(sin(radians), -cos(radians));
		unitVectors[i].X() = sin(radians);
		unitVectors[i].Y() =-cos(radians);
	}
	return unitVectors;
}
static const Point* Angle::unitCache = BuildUnitCache();
*/

alignas(64) static struct {
    alignas(16) double x;
    double y;
} unitVectors[STEPS];

static constexpr const Point* BuildUnitCache()
{
	for(int i = 0; i < STEPS; ++i)
	{
		double radians = i * STEP_TO_RAD;
		// The graphics use the usual screen coordinate system, meaning that
		// positive Y is down rather than up. Angles are clock angles, i.e.
		// 0 is 12:00 and angles increase in the clockwise direction. So, an
		// angle of 0 degrees is pointing in the direction (0, -1).
//		unitVectors[i] = { sin(radians), -cos(radians) };
//		unitVectors[i] = Point(sin(radians), -cos(radians));
		unitVectors[i].x =  sin(radians);
		unitVectors[i].y = -cos(radians);
	}
//	return unitVectors;
	return (const Point*) unitVectors;
}
const Point* Angle::unitCache = BuildUnitCache();

const Point &Angle::Unit() const
{
//	return *reinterpret_cast<Point*>(&unitVectors[angle]);  // violates strict aliasing
	//return Point(unitVectors[angle].x, unitVectors[angle].y);
	return unitCache[angle];
}


static vector<Point> local_cache;
Point Angle_LUT(int32_t idx) {
	return local_cache[idx];
}

const Point & Angle_LUT_byref(int32_t idx) {
	return local_cache[idx];
}


// Convert an angle back to a value in degrees.
double Angle::Degrees() const
{
	// Most often when this function is used, it's in settings where it makes
	// sense to return an angle in the range [-180, 180) rather than in the
	// Angle's native range of [0, 360).
	return angle * (1. / DEG_TO_STEP)  -  360. * (angle >= STEPS / 2);
}
