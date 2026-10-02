-- Shared helpers for the OpenAdobe Ardour drivers (run with ardour8-lua; see engines/ardour.py).
-- Any uncaught Lua error aborts luasession with "FATAL: exception not rethrown", so every driver
-- runs its body through `main`, which reports the message on stderr and ends with a failing status.
io.stdout:setvbuf("no")

function main(body, ...)
	local ok, err = xpcall(body, debug.traceback, ...)
	if not ok then
		io.stderr:write("OA-ERROR: " .. tostring(err) .. "\n")
		pcall(close_session)
		error("OA-FAILED", 0) -- os.exit is not available; an error at top level gives exit status 1
	end
	pcall(close_session) -- leaving with an open session aborts luasession
	print("OA-DONE")
end

function route_list(s)
	local out = {}
	for r in s:get_routes():iter() do out[#out + 1] = r end
	return out
end

NO_GROUP = PBD.GroupControlDisposition.NoGroup

-- Ardour applies many control and connection changes asynchronously on the engine thread; poll until `check`
-- holds so that the following step (a save, an export) sees the new state.
function settle(what, check)
	for _ = 1, 500 do
		if check() then return end
		ARDOUR.LuaAPI.usleep(10000)
	end
	error("timed out waiting for " .. what)
end

function set_control(ctl, value, what)
	ctl:set_value(value, NO_GROUP)
	settle(what, function() return math.abs(ctl:get_value() - value) < 1e-6 end)
end

-- Where a route's output goes: the names of the routes whose inputs its ports are connected to
function destinations(s, route)
	local dests = {}
	for _, other in ipairs(route_list(s)) do
		if other:name() ~= route:name() and not other:input():isnil() then
			local all = true
			for i = 0, 1 do
				all = all and route:output():audio(i):connected_to(other:input():audio(i):name())
			end
			if all then dests[#dests + 1] = other:name() end
		end
	end
	return dests
end
