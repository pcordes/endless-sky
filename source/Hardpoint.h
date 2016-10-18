/* Hardpoint.h
Copyright (c) 2016 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*/

#ifndef HARDPOINT_H_
#define HARDPOINT_H_

#include "Point.h"
#include "Angle.h"
#include "Outfit.h"

#include <list>

class Effect;
class Outfit;
class Projectile;
class Ship;



// A single weapon hardpoint on the ship (i.e. a gun port or turret mount),
// which may or may not have a weapon installed.
class Hardpoint {
public:
	// Constructor. Hardpoints may or may not specify what weapon is in them.
	Hardpoint(const Point &point, bool isTurret, const Outfit *outfit = nullptr);
	
	// Get the weapon installed in this hardpoint (or null if there is none).
	const Outfit *GetOutfit() const;
	// Get the location, relative to the center of the ship, from which
	// projectiles of this weapon should originate. This point must be
	// rotated to take the ship's current facing direction into account.
	const Point &GetPoint() const;
	// Get the convergence angle adjustment of this weapon (if it's a gun).
	const Angle &GetAngle() const;
	// Shortcuts for querying weapon characteristics.
	bool IsTurret() const;
	bool IsHoming() const;
	bool IsAntiMissile() const;
	
	// Check if this weapon is ready to fire.
	bool IsReady() const;
	// Check if this weapon was firing in the previous step.
	bool WasFiring() const;
	// If this is a burst weapon, get the number of shots left in the burst.
	int BurstRemaining() const;
	// Perform one step (i.e. decrement the reload count).
	void Step();
	
	// Fire this weapon. If it is a turret, it automatically points toward
	// the given ship's target. If the weapon requires ammunition, it will
	// be subtracted from the given ship.
	void Fire(Ship &ship, std::list<Projectile> &projectiles, std::list<Effect> &effects);
	// Fire an anti-missile. Returns true if the missile should be killed.
	bool FireAntiMissile(Ship &ship, const Projectile &projectile, std::list<Effect> &effects);
	
	// Install a weapon here (assuming it is empty). This is only for
	// Armament to call internally.
	void Install(const Outfit *outfit);
	// Uninstall the outfit from this port (if it has one).
	void Uninstall();
	
private:
	// Reset the reload counters and expend ammunition, if any.
	void Fire(Ship &ship, const Point &start, const Angle &aim);
	
private:
	// The weapon installed in this hardpoint.
	const Outfit *outfit = nullptr;
	// Hardpoint location, in world coordinates relative to the ship's center.
	Point point;
	// Angle adjustment for convergence.
	Angle angle;
	// Reload timers and other attributes.
	double reload = 0.;
	double burstReload = 0.;
	int burstCount = 0;
	bool isTurret = false;
	bool isFiring = false;
	bool wasFiring = false;
};

// Get the weapon in this hardpoint. This returns null if there is none.
inline const Outfit *Hardpoint::GetOutfit() const { return outfit; }

// Get the location, relative to the center of the ship, from which
// projectiles of this weapon should originate.
inline const Point &Hardpoint::GetPoint() const { return point; }

// Get the convergence angle adjustment of this weapon (guns only, not turrets).
inline const Angle &Hardpoint::GetAngle() const { return angle; }

// Find out if this is a turret hardpoint (whether or not it has a turret
// installed).
inline bool Hardpoint::IsTurret() const { return isTurret; }

// Find out if this hardpoint has a homing weapon installed.
inline bool Hardpoint::IsHoming() const
{
	return outfit && outfit->Homing();
}

// Find out if this hardpoint has an anti-missile installed.
inline bool Hardpoint::IsAntiMissile() const
{
	return outfit && outfit->AntiMissile() > 0;
}

// Check if this weapon is ready to fire.
inline bool Hardpoint::IsReady() const
{
	return outfit && burstReload <= 0. && burstCount;
}

// Check if this weapon fired the last time it was able to fire. This is to
// figure out if the stream spacing timer should be applied or not.
inline bool Hardpoint::WasFiring() const { return wasFiring; }

// Get the number of remaining burst shots before a full reload is required.
inline int Hardpoint::BurstRemaining() const { return burstCount; }

#endif
