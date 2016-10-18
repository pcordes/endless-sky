/* Mask.h
Copyright (c) 2014 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#ifndef MASK_H_
#define MASK_H_

#include "Angle.h"
#include "Point.h"

#include <vector>

class ImageBuffer;



// Class representing the outline of an object, with functions for checking if a
// line segment intersects that object or if a point is within a certain distance.
// The outline is represented in polygonal form, which allows intersection tests
// to be done much more efficiently than if we were testing individual pixels in
// the image itself.
class Mask {
public:
	// Default constructor.
	Mask();
	
	// Construct a mask from the alpha channel of an image.
	void Create(const ImageBuffer *image);
	
	// Check whether a mask was successfully loaded.
	bool IsLoaded() const;
	
	// Check if this mask intersects the given line segment (from sA to vA). If
	// it does, return the fraction of the way along the segment where the
	// intersection occurs. The sA should be relative to this object's center.
	// If this object contains the given point, the return value is 0. If there
	// is no collision, the return value is 1.
	double Collide(Point sA, Point vA, Angle facing) const;
	
	// Check whether the mask contains the given point.
	bool Contains(Point point, Angle facing) const;
	
	// Find out whether this object (rotated and scaled as represented by the
	// given unit vector) is within the given range of the given point.
	bool WithinRange(Point point, Angle facing, double range) const;
	
	// Find out how close the given point is to the mask.
	double Range(Point point, Angle facing) const;
	
	static const Mask& EmptyMask();
	static const Mask emptymask;
private:
public:
	double Intersection(Point sA, Point vA) const;
	bool Contains(Point point) const;
	
	
private:
	// prevents using an unaligned load to get next or previous when we need next.x - prev.x
	struct xy_interleave {
		static constexpr unsigned vecSize = 4;
		static constexpr unsigned alignMask = vecSize-1;
		alignas(64) float x[vecSize];
		float dx[vecSize]; // current - prev;   prev.x = x-dx
		float y[vecSize];
		float dy[vecSize];
	};
	std::vector<xy_interleave> outline_simd;
	std::vector<Point> outline;
	double radius;
public:
	size_t OutlineCount() const { return outline.size(); }
	double GetRadius() const { return radius; }
	const std::vector<Point> &Outline() const { return outline; }
};


// Check whether a mask was successfully loaded.  inline to save code size
inline bool Mask::IsLoaded() const {
	return !outline.empty();
}
inline const Mask& Mask::EmptyMask() { return emptymask; }

#endif
