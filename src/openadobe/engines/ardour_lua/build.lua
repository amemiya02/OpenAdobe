-- Skeleton of a session: tracks, one bus per role, routing, faders, mute, plugin chains.
-- Regions are added afterwards by engines/ardour.py (sources are not creatable from Lua headlessly).
-- usage: ardour8-lua build.lua <common.lua> <spec.lua>   (script arguments arrive in the `arg` table)
local common, specfile = arg[1], arg[2]
dofile(common)

-- Plugins go in front of the fader, in order. add_processor_by_index counts in the editor's numbering (hidden
-- processors such as the disk reader and trim do not count), so index k is the k-th slot before the fader.
local function add_plugins(s, route, list, report)
	for k, fx in ipairs(list) do
		local ptype = fx.type == "lua" and ARDOUR.PluginType.Lua or ARDOUR.PluginType.LV2
		local p = ARDOUR.LuaAPI.new_plugin(s, fx.id, ptype, "")
		if p:isnil() then error("plugin not found: " .. fx.id) end
		route:add_processor_by_index(p, k - 1, nil, true)
		for idx, val in pairs(fx.params or {}) do
			ARDOUR.LuaAPI.set_processor_param(p, idx, val)
		end
		report[#report + 1] = route:name() .. ": " .. fx.id
	end
end

local function db(x) return ARDOUR.DSP.dB_to_coefficient(x) end

main(function()
	local spec = dofile(specfile)
	local s = create_session(spec.dir, spec.name, spec.rate)
	assert(s ~= nil, "create_session failed (does the session already exist?)")
	local master = s:master_out()
	assert(not master:isnil(), "no master bus")

	-- create every route first; Ardour auto-connects new tracks to Master asynchronously, so the routing is
	-- only fixed up afterwards (otherwise that late auto-connect leaks a track straight into Master)
	local buses, tracks = {}, {}
	for _, role in ipairs(spec.buses) do
		local bl = s:new_audio_route(2, 2, nil, 1, spec.bus_names[role], ARDOUR.PresentationInfo.Flag.AudioBus,
			ARDOUR.PresentationInfo.max_order)
		buses[role] = bl:front()
	end
	for _, t in ipairs(spec.tracks) do
		local tl = s:new_audio_track(2, 2, nil, 1, t.name, ARDOUR.PresentationInfo.max_order,
			ARDOUR.TrackMode.Normal, true)
		tracks[#tracks + 1] = tl:front()
		tracks[#tracks]:set_strict_io(true)
	end
	local report = {}
	for i, t in ipairs(spec.tracks) do
		local tr = tracks[i]
		set_control(tr:gain_control(), db(t.gain_db), "fader " .. t.name)
		if t.muted then set_control(tr:mute_control(), 1, "mute " .. t.name) end
		add_plugins(s, tr, t.fx, report)
	end
	set_control(master:gain_control(), db(spec.master_db), "master fader")
	add_plugins(s, master, spec.master_fx, report)

	-- Routing comes last: Ardour auto-connects new tracks to Master asynchronously, and again whenever a plugin
	-- changes a route's channel configuration, which would leak a track straight into Master next to its bus.
	for _, tr in ipairs(tracks) do
		pcall(settle, "auto-connect of " .. tr:name(), function()
			for _, d in ipairs(destinations(s, tr)) do
				if d == master:name() then return true end
			end
			return false
		end)
	end
	ARDOUR.LuaAPI.usleep(300000)
	local function route_to(src, dst)
		src:output():disconnect_all(nil)
		for i = 0, 1 do
			assert(src:output():audio(i):connect(dst:input():audio(i):name()) == 0, "connect " .. src:name())
		end
		settle("routing of " .. src:name(), function()
			local d = destinations(s, src)
			return #d == 1 and d[1] == dst:name()
		end)
	end
	for _, role in ipairs(spec.buses) do route_to(buses[role], master) end
	for i, t in ipairs(spec.tracks) do route_to(tracks[i], buses[t.role] or master) end
	-- and make sure nothing moves again
	ARDOUR.LuaAPI.usleep(500000)
	for i, t in ipairs(spec.tracks) do
		local d = destinations(s, tracks[i])
		assert(#d == 1 and d[1] == (buses[t.role] or master):name(), "routing of " .. t.name .. " changed afterwards")
	end
	s:save_state("")
	for _, line in ipairs(report) do print("OA-PLUGIN " .. line) end
end)
