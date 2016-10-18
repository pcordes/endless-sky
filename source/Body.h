/* Body.h
Copyright (c) 2016 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#ifndef BODY_H_
#define BODY_H_

#include <cstdint>
#include <string>
#include <algorithm>

#include "Angle.h"
#include "Point.h"

class DataNode;
class DataWriter;
class Government;
class Mask;
class Sprite;



// Class representing any object in the game that has a position, velocity, and
// facing direction and usually also has a sprite.
class Body {
public:
	// Class representing the current animation state, which may be halfway in
	// between two frames.
	class Frame {
	public:
		uint32_t first = 0;
		uint32_t second = 0;
		float fade = 0.f;
	};
	
	
public:
	// Constructors.
	Body() = default;
	Body(const Sprite *sprite, Point position, Point velocity = Point(), Angle facing = Angle(), double zoom = 1.);
	Body(const Body &sprite, Point position, Point velocity = Point(), Angle facing = Angle(), double zoom = 1.);
	
	// Check that this Body has a sprite and that the sprite has at least one frame.
	bool HasSprite() const;
	// Access the underlying Sprite object.
	const Sprite *GetSprite() const;
	// Get the dimensions of the sprite.
	double Width() const;
	double Height() const;
	// Get the farthest a part of this sprite can be from its center.
	double Radius() const;
	// Which color swizzle should be applied to the sprite?
	int GetSwizzle() const;
	// Get the sprite and mask for the given time step, or the current step
	Frame GetFrame(int step) const;
	Frame GetFrame() const;
	const Mask &GetMask(int step) const;
	const Mask &GetMask() const;
	
	// Positional attributes.
	const Point &Position() const;
	const Point &Velocity() const;
	const Angle &Facing() const;
	Point Unit() const;
	double Zoom() const;
	
	// Store the government here too, so that collision detection that is based
	// on the Body class can figure out which objects will collide.
	const Government *GetGovernment() const;
	
	// Sprite serialization.
	void LoadSprite(const DataNode &node);
	void SaveSprite(DataWriter &out) const;
	// Set the sprite.
	void SetSprite(const Sprite *sprite);
	// Set the color swizzle.
	void SetSwizzle(int swizzle);
	
	
protected:
	// Adjust the frame rate.
	void SetFrameRate(double framesPerSecond);
	void AddFrameRate(double framesPerSecond);
	
	
protected:
	// Basic positional attributes.
	Point position;
	Point velocity;
	Angle angle;
	// A zoom of 1 means the sprite should be drawn at half size. For objects
	// whose sprites should be full size, use zoom = 2.
	double zoom = 1.;
	
	// Government, for use in collision checks.
	const Government *government = nullptr;
	
	
private:
	// Set what animation step we're on. This affects future calls to GetMask()
	// and GetFrame().
	void SetStep(int step) const;
	
	
private:
	// Animation parameters.
	const Sprite *sprite = nullptr;
	// Allow objects based on this one to adjust their frame rate and swizzle.
	int swizzle = 0;
	
	float frameRate = 2.f / 60.f;
	int delay = 0;
	// The chosen frame will be (step * frameRate) + frameOffset.
	mutable float frameOffset = 0.f;
	mutable bool startAtZero = false;
	mutable bool randomize = false;
	bool repeat = true;
	bool rewind = false;
	
	// Frame info for the current step:
	mutable int currentStep = -1;
	mutable const Mask *mask = nullptr;
	mutable Frame frame;
};



// Constructor, based on a Sprite.
inline Body::Body(const Sprite *sprite, Point position, Point velocity, Angle facing, double zoom)
	: position(position), velocity(velocity), angle(facing), zoom(zoom), sprite(sprite), randomize(true)
{
}

// Constructor, based on the animation from another Body object.
inline Body::Body(const Body &sprite, Point position, Point velocity, Angle facing, double zoom)
{
	*this = sprite;
	this->position = position;
	this->velocity = velocity;
	this->angle = facing;
	this->zoom = zoom;
}

#include "Sprite.h"

// Check that this Body has a sprite and that the sprite has at least one frame.
inline bool Body::HasSprite() const
{
	return (sprite && sprite->Frames());
}

// Access the underlying Sprite object.
inline const Sprite *Body::GetSprite() const { return sprite; }

// Get the width of this object, in world coordinates (i.e. taking zoom into account).
inline double Body::Width() const
{
	return sprite ? (.5 * zoom) * sprite->Width() : 0.;
}

// Get the height of this object, in world coordinates (i.e. taking zoom into account).
inline double Body::Height() const
{
	return sprite ? (.5 * zoom) * sprite->Height() : 0.;
}

// Get the farthest a part of this sprite can be from its center.
inline double Body::Radius() const
{
	return .5 * Point(Width(), Height()).Length();
}


// Which color swizzle should be applied to the sprite?
inline int Body::GetSwizzle() const { return swizzle; }

// the (int step) versions are probably better not to inline
inline Body::Frame Body::GetFrame() const { return frame; }
inline const Mask &Body::GetMask() const
{
	return (mask ? *mask : Mask::emptymask);
}

// Position, in world coordinates (zero is the system center).
inline const Point &Body::Position() const { return position; }
// Velocity, in pixels per second.
inline const Point &Body::Velocity() const { return velocity; }
// Direction this Body is facing in.
inline const Angle &Body::Facing() const { return angle; }

// Unit vector in the direction this body is facing. This represents the scale
// and transform that should be applied to the sprite before drawing it.
inline Point Body::Unit() const
{
	return angle.Unit() * (.5 * Zoom());
}

// Zoom factor. THis controls how big the sprite should be drawn.
inline double Body::Zoom() const {
	return std::max(zoom, 0.);
}

// Store the government here too, so that collision detection that is based
// on the Body class can figure out which objects will collide.
inline const Government *Body::GetGovernment() const { return government; }

// Set the sprite.
inline void Body::SetSprite(const Sprite *sprite) {  this->sprite = sprite;  }
// Set the color swizzle.
inline void Body::SetSwizzle(int swizzle) { this->swizzle = swizzle; }



// Set the frame rate of the sprite. This is used for objects that just specify
// a sprite instead of a full animation data structure.
inline void Body::SetFrameRate(double framesPerSecond) {
	frameRate = framesPerSecond * (1. / 60.);
}

// Add the given amount to the frame rate.
inline void Body::AddFrameRate(double framesPerSecond)
{
	frameRate += framesPerSecond * (1. / 60.);
}

#endif
