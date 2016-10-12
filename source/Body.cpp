/* Body.h
Copyright (c) 2016 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#include "Body.h"

#include "DataNode.h"
#include "DataWriter.h"
#include "Random.h"
#include "Sprite.h"
#include "SpriteSet.h"

#include <algorithm>
#include <cmath>

using namespace std;


// Load the sprite specification, including all animation attributes.
void Body::LoadSprite(const DataNode &node)
{
	if(node.Size() < 2)
		return;
	sprite = SpriteSet::Get(node.Token(1));
	
	// The only time the animation does not start on a specific frame is if no
	// start frame is specified and it repeats. Since a frame that does not
	// start at zero starts when the game started, it does not make sense for it
	// to do that unless it is repeating endlessly.
	for(const DataNode &child : node)
	{
		if(child.Token(0) == "frame rate" && child.Size() >= 2 && child.Value(1) >= 0.)
			frameRate = child.Value(1) * (1. / 60.);
		else if(child.Token(0) == "frame time" && child.Size() >= 2 && child.Value(1) > 0.)
			frameRate = 1. / child.Value(1);
		else if(child.Token(0) == "delay" && child.Size() >= 2 && child.Value(1) > 0.)
			delay = child.Value(1);
		else if(child.Token(0) == "start frame" && child.Size() >= 2)
		{
			frameOffset += child.Value(1);
			startAtZero = true;
		}
		else if(child.Token(0) == "random start frame")
			randomize = true;
		else if(child.Token(0) == "no repeat")
		{
			repeat = false;
			startAtZero = true;
		}
		else if(child.Token(0) == "rewind")
			rewind = true;
		else
			child.PrintTrace("Skipping unrecognized attribute:");
	}
}



// Save the sprite specification, including all animation attributes.
void Body::SaveSprite(DataWriter &out) const
{
	if(!sprite)
		return;
	
	out.Write("sprite", sprite->Name());
	out.BeginChild();
	{
		if(frameRate != static_cast<float>(2. / 60.))
			out.Write("frame rate", frameRate * 60.);
		if(delay)
			out.Write("delay", delay);
		if(randomize)
			out.Write("random start frame");
		if(!repeat)
			out.Write("no repeat");
		if(rewind)
			out.Write("rewind");
	}
	out.EndChild();
}

// Set the current time step.
void Body::SetStep(int step) const
{
	// If the step is negative or there is no sprite, do nothing. This updates
	// and caches the mask and the frame so that if further queries are made at
	// this same time step, we don't need to redo the calculations.
	if(step < 0 || !sprite || !sprite->Frames() || step == currentStep)
		return;
	currentStep = step;
	
	// If the sprite only has one frame, no need to animate anything.
	int frames = sprite->Frames();
	if(frames <= 1)
	{
		frame.first = sprite->Texture();
		mask = &sprite->GetMask();
		return;
	}
	
	// If this is the very first step, fill in some values that we could not set
	// until we knew the sprite's frame count and the starting step.
	if(randomize)
	{
		randomize = false;
		frameOffset += Random::Int(sprite->Frames() + delay);
	}
	else if(startAtZero)
	{
		startAtZero = false;
		// Adjust frameOffset so that this step's frame is exactly 0 (no fade).
		frameOffset -= frameRate * step;
	}
	
	// Figure out what fraction of the way in between frames we are. Avoid any
	// possible floating-point glitches that might result in a negative frame.
	float frameF = max(0.f, frameRate * step + frameOffset);
	frame.fade = modf(frameF, &frameF);
	step = frameF;
	
	int firstIndex = 0;
	int secondIndex = 0;
	if(!rewind)
	{
		// This is not a rewinding sprite. It is either playing once through, or
		// looping from start to end over and over.
		if(!repeat && step >= frames - 1)
		{
			// If the sprite is not repeating and it has reached the end, just
			// stay on the last frame.
			firstIndex = frames - 1;
			frame.fade = 0.f;
		}
		else
		{
			// Figure out which animation step we're on. If the sprite has a
			// delay and all the non-delay frames have been shown, show the
			// first frame until the delay is over.
			step %= (frames + delay);
			if(step >= frames)
				frame.fade = 0.f;
			else
			{
				// The Sprite class automatically wraps a step that is out of
				// bounds, so if steps == frames, (steps + 1) maps to frame 0.
				firstIndex = step;
				secondIndex = step + 1;
			}
		}
	}
	else
	{
		// This sprite "rewinds" - that is, it plays forward, then in reverse,
		// then possibly delays for some number of frames, then may repeat.
		if(!repeat && step >= 2 * (frames - 1))
			frame.fade = 0.f;
		else
		{
			// The first and last frame only get shown for one step. All the
			// others get shown twice, and a delay may also be added, so:
			step %= 2 * (frames - 1) + delay;
			if(step < frames - 1)
			{
				firstIndex = step;
				secondIndex = step + 1;
			}
			else if(step < 2 * (frames - 1))
			{
				step = 2 * (frames - 1) - step;
				firstIndex = step;
				secondIndex = step - 1;
			}
			else
				frame.fade = 0.f;
		}
	}
	// Cache the frame and mask info so as long as the step stays the same, none
	// of the above calculations need to be redone. This is important for objects
	// whose masks may be queried many times for collision tests.
	frame.first = sprite->Texture(firstIndex);
	frame.second = sprite->Texture(secondIndex);
	mask = &sprite->GetMask(frame.fade > .5f ? secondIndex : firstIndex);
}
