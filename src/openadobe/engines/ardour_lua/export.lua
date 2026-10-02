local common, specfile = arg[1], arg[2]
dofile(common)

main(function()
	local spec = dofile(specfile)
	local s = load_session(spec.dir, spec.name)
	assert(s ~= nil, "load_session failed")
	for _, r in ipairs(route_list(s)) do
		local tr = r:to_track()
		if not tr:isnil() then
			local n = 0
			for reg in tr:playlist():region_list():iter() do
				n = n + 1
				print(string.format("OA-REGION %s %s pos=%d len=%d start=%d", r:name(), reg:name(),
					reg:position():samples(), reg:length():samples(), reg:start():samples()))
			end
			print(string.format("OA-TRACK %s regions=%d gain=%.4f muted=%d to=%s", r:name(), n,
				r:gain_control():get_value(), r:mute_control():get_value(), table.concat(destinations(s, r), ",")))
		elseif not r:is_master() and not r:is_monitor() then
			print(string.format("OA-BUS %s to=%s", r:name(), table.concat(destinations(s, r), ",")))
		end
	end
	for _, job in ipairs(spec.jobs) do
		-- a bus that is "off" gets its fader pulled to -inf, the others are at unity: what reaches Master is then
		-- exactly the output of the remaining buses
		for _, r in ipairs(route_list(s)) do
			if job.off[r:name()] ~= nil then
				set_control(r:gain_control(), job.off[r:name()] and 0 or 1, "bus " .. r:name())
			end
		end
		local e = s:simple_export()
		e:set_name(job.name)
		e:set_folder(spec.out_dir)
		e:set_range(spec.start, spec.stop)
		e:set_preset(spec.preset)
		print("OA-CHECK", e:check_outputs())
		print("OA-EXPORT", job.name, e:run_export())
	end
end)
